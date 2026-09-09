import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from image_context import ImageUnavailable, normalize_image
from ocr_context import OCRImageContext, trim_image_context
from ocr_engine import LocalOCR, excerpt, parse_tsv


def picture(size=(80, 50), color='white'):
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format='PNG')
    return output.getvalue()


class OCRParsingTests(unittest.TestCase):
    def test_tsv_preserves_numbers_uncertainty_and_visual_rows(self):
        tsv = 'level\tleft\ttop\twidth\theight\tconf\ttext\n'
        tsv += '5\t10\t10\t40\t20\t91\t合计\n5\t100\t10\t50\t20\t49\t123.45\n'
        rows = parse_tsv(tsv)
        self.assertEqual(rows[0]['text'], '合计 | 123.45')
        self.assertIn('[存疑]', excerpt({'lines': rows}))
        self.assertIn('123.45', excerpt({'lines': rows}))

    def test_invalid_tsv_rows_are_ignored(self):
        tsv = 'level\tleft\ttop\twidth\theight\tconf\ttext\n5\tbad\t0\t1\t1\t80\tx\n'
        tsv += '5\t0\t0\t1\t1\tnan\ty\n'
        self.assertEqual(parse_tsv(tsv), [])

    def test_keyword_question_selects_late_row_without_rewriting(self):
        rows = [{'text': '普通记录内容' * 10, 'confidence': 90, 'y': i * 20} for i in range(60)]
        rows.append({'text': '订单编号 AB-2026-009 合计 123.45', 'confidence': 90, 'y': 2000})
        value = excerpt({'lines': rows}, '订单编号是什么', 600)
        self.assertIn('AB-2026-009', value)
        self.assertIn('部分原文', value)
        self.assertLessEqual(len(value), 600)

    def test_empty_is_not_claimed_as_no_text(self):
        self.assertIn('未取得', excerpt({'status': 'empty', 'lines': []}))

    def test_oversized_input_does_not_launch_worker(self):
        self.assertEqual(LocalOCR().extract(b'')['status'], 'unavailable')

    def test_media_budget_trim_preserves_sender_and_declares_omissions(self):
        source = '\n'.join(json.dumps({'speaker_key': str(i), 'description': '内容' * 2000}, ensure_ascii=False) for i in range(3))
        result = trim_image_context(source, 1500)
        self.assertLessEqual(len(result), 1500)
        rows = [json.loads(line) for line in result.splitlines()]
        self.assertEqual([r['speaker_key'] for r in rows], ['0', '1', '2'])
        self.assertTrue(all('省略' in r['description'] for r in rows))
        escaped = json.dumps({'speaker_key': 'a', 'description': '\n"\\' * 3000})
        self.assertLessEqual(len(trim_image_context(escaped, 1000)), 1000)


class OCRStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = sqlite3.connect(':memory:')
        self.ai = Mock()
        self.ai.describe_image.return_value = ('一张订单截图', {})
        self.images = OCRImageContext(self.root, self.state, self.ai, 'bot')
        self.images.ocr = Mock(revision='test-1')
        self.images.ocr.extract.return_value = {'status': 'ok', 'notes': [],
            'lines': [{'text': '订单编号 AB123 金额 99.80', 'confidence': 90, 'y': 10}]}
        self.account = self.root / 'account'
        self.folder = self.account / 'msg/attach' / hashlib.md5(b'111@chatroom').hexdigest() / '2026-09/Img'
        self.folder.mkdir(parents=True)
        self.path = self.folder / ('a' * 32 + '.dat')
        self.path.write_bytes(picture())
        self.item = {'local_id': 1, 'create_time': 1788520000, 'packed_info_data': b'a' * 32}

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def describe(self, **kwargs):
        return self.images.describe(self.account, '111@chatroom', 'message/0', self.item, **kwargs)

    def test_raw_ocr_and_vision_summary_both_reach_reply(self):
        text = self.describe()
        self.assertIn('一张订单截图', text)
        self.assertIn('AB123', text)
        self.assertEqual(self.images.ocr.extract.call_args.args[0], self.path.read_bytes())
        self.assertIn('不可信引用', text)

    def test_old_vision_cache_gets_ocr_without_repeating_vision(self):
        normalized, _ = normalize_image(self.path.read_bytes())
        self.state.execute('INSERT INTO image_memory VALUES(?,?,?,?,?)',
            ('111@chatroom', 'message/0', 1, hashlib.sha256(normalized).hexdigest(), '旧摘要'))
        self.state.commit()
        self.assertIn('AB123', self.describe())
        self.ai.describe_image.assert_not_called()
        self.images.ocr.extract.assert_called_once()

    def test_repeated_question_uses_both_caches_and_no_cross_group(self):
        self.describe()
        self.describe(question='订单编号是什么')
        self.ai.describe_image.assert_called_once()
        self.images.ocr.extract.assert_called_once()
        self.assertIsNone(self.images.cached('222@chatroom', 'message/0', 1))
        with self.assertRaises(ImageUnavailable):
            self.images.describe(self.account, '222@chatroom', 'message/0', self.item)

    def test_vision_failure_keeps_ocr(self):
        self.ai.describe_image.side_effect = RuntimeError('vision failure')
        value = self.describe()
        self.assertIn('AB123', value)
        self.assertIn('视觉模型本次未成功', value)

    def test_ocr_failure_keeps_vision_and_retry_is_bounded(self):
        self.images.ocr.extract.return_value = {'status': 'timeout', 'lines': [], 'notes': ['OCR 超时']}
        self.assertIn('一张订单截图', self.describe())
        self.assertIn('OCR 超时', self.describe())
        self.images.ocr.extract.assert_called_once()

    def test_both_fail_do_not_claim_image_observed(self):
        self.ai.describe_image.side_effect = RuntimeError('vision failure')
        self.images.ocr.extract.return_value = {'status': 'unavailable', 'lines': []}
        with self.assertRaises(ImageUnavailable):
            self.describe()

    def test_higher_resolution_source_refreshes_ocr_cache(self):
        self.describe()
        self.path.write_bytes(picture((200, 200)))
        self.describe()
        self.assertEqual(self.images.ocr.extract.call_count, 2)

    def test_ocr_revision_change_refreshes_only_ocr(self):
        self.describe()
        self.images.ocr.revision = 'test-2'
        self.describe()
        self.assertEqual(self.images.ocr.extract.call_count, 2)
        self.ai.describe_image.assert_called_once()

    def test_changed_image_with_failed_vision_does_not_reuse_old_scene(self):
        self.describe()
        self.path.write_bytes(picture((200, 200), 'blue'))
        self.ai.describe_image.side_effect = RuntimeError('vision failure')
        self.describe()
        self.assertNotIn('一张订单截图', self.images.cached('111@chatroom', 'message/0', 1))

    def test_disabling_ocr_leaves_original_vision_behavior(self):
        self.images.ocr = None
        self.assertEqual(self.describe(), '一张订单截图')

    def test_original_unavailable_does_not_invent_ocr(self):
        self.state.execute('INSERT INTO image_memory VALUES(?,?,?,?,?)',
            ('111@chatroom', 'message/0', 1, 'old', '旧摘要'))
        self.state.commit()
        self.path.unlink()
        value = self.describe()
        self.assertIn('尚未补做OCR', value)
        self.assertIn('旧摘要', value)

    def test_context_size_bounded(self):
        self.images.ocr.extract.return_value['lines'] *= 200
        self.assertLessEqual(len(self.describe(max_chars=800)), 800)
        self.assertLessEqual(len(self.images.cached('111@chatroom', 'message/0', 1)), 1100)


if __name__ == '__main__':
    unittest.main()
