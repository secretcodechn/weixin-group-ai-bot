"""Add independent OCR evidence to the existing image/sticker pipeline."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

from image_context import ImageContext, ImageUnavailable, MAX_BYTES, decode_dat, normalize_image
from ocr_engine import LocalOCR, excerpt


def trim_image_context(value, limit):
    """Last-resort context budget guard; preserve author metadata and disclose omissions."""
    notice = '图片资料因上下文长度限制有所省略；不能据此断言未列出的内容不存在，请按字段或裁图追问。'
    if len(value) <= limit:
        return value
    try:
        records = [json.loads(line) for line in value.splitlines()]
        if not records or not all(isinstance(r, dict) and 'description' in r for r in records):
            raise ValueError('Not attributed media records')
        for record in records:
            record['description'] = ''
        overhead = len('\n'.join(json.dumps(r, ensure_ascii=False) for r in records))
        allowance = (limit - overhead) // len(records) - len(notice) - 12
        if allowance < 80:
            return notice[:limit]
        originals = [json.loads(line) for line in value.splitlines()]
        low, high, best = 0, allowance, notice[:limit]
        while low <= high:
            size = (low + high) // 2
            for record, original in zip(records, originals):
                record['description'] = str(original['description'])[:size] + '\n' + notice
            candidate = '\n'.join(json.dumps(r, ensure_ascii=False) for r in records)
            if len(candidate) <= limit:
                best, low = candidate, size + 1
            else:
                high = size - 1
        return best
    except (ValueError, TypeError):
        return value[:max(0, limit - len(notice) - 1)] + '\n' + notice[:limit]


class OCRImageContext(ImageContext):
    def __init__(self, root, state, ai, bot_id, *, ocr_enabled=True):
        super().__init__(root, state, ai, bot_id)
        self.ocr = LocalOCR() if ocr_enabled else None
        state.execute('''CREATE TABLE IF NOT EXISTS image_ocr_memory(
            group_id TEXT,shard TEXT,local_id INTEGER,source_hash TEXT,revision TEXT,
            result TEXT,updated INTEGER,PRIMARY KEY(group_id,shard,local_id))''')
        state.commit()

    def ocr_record(self, group, shard, ident):
        row = self.state.execute('SELECT source_hash,revision,result,updated FROM image_ocr_memory '
            'WHERE group_id=? AND shard=? AND local_id=?', (group, shard, ident)).fetchone()
        if row is None:
            return None
        return {'source_hash': row[0], 'revision': row[1], 'result': json.loads(row[2]), 'updated': row[3]}

    @staticmethod
    def combine(vision, result, question='', max_chars=2200):
        if result is None:
            return vision or ''
        # OCR does not pass through the vision model's short summary a second time.
        overview = '画面理解：' + (vision or '视觉模型本次未成功，仅能依据OCR文字，不能推测画面。')
        overview = overview[:min(650, max_chars // 3)]
        return (overview + '\n' + excerpt(result, question, max(150, max_chars - len(overview) - 1)))[:max_chars]

    def cached(self, group_id, shard, local_id):
        vision = super().cached(group_id, shard, local_id)
        record = self.ocr_record(group_id, shard, local_id) if self.ocr else None
        if record and record['revision'] == self.ocr.revision:
            stored = self.state.execute('SELECT image_hash FROM image_memory WHERE group_id=? AND shard=? AND local_id=?',
                                       (group_id, shard, local_id)).fetchone()
            if record['result'].get('vision_hash') and (not stored or stored[0] != record['result']['vision_hash']):
                vision = None
            return self.combine(vision, record['result'], max_chars=1100)
        return vision

    def describe(self, account, group_id, shard, item, *, question='', max_chars=2200):
        if not self.ocr:
            return super().describe(account, group_id, shard, item)
        ident = item['local_id']
        old_vision = self.state.execute('SELECT image_hash,summary FROM image_memory '
            'WHERE group_id=? AND shard=? AND local_id=?', (group_id, shard, ident)).fetchone()
        record = self.ocr_record(group_id, shard, ident)
        last_error = ImageUnavailable('这张图片尚未下载到本机')
        for path in self.paths(account, group_id, item):
            try:
                with path.open('rb') as f:
                    data = f.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ImageUnavailable('图片超过大小限制')
                key, xor = self.keys(account, data) if data[:6] == b'\x07\x08V2\x08\x07' else (None, None)
                sticker = (item.get('local_type', 3) & 0xffffffff) == 47
                decoded = (data if path.suffix == '.jpg' else self.decode_sticker(data, item)) if sticker else decode_dat(data, key, xor)
                image, mime = normalize_image(decoded)
            except (ImageUnavailable, OSError) as exc:
                last_error = exc
                continue
            source_hash, vision_hash = hashlib.sha256(decoded).hexdigest(), hashlib.sha256(image).hexdigest()
            vision = old_vision[1] if old_vision and old_vision[0] == vision_hash else None
            result = None
            if record and record['source_hash'] == source_hash and record['revision'] == self.ocr.revision:
                cached_result = record['result']
                if cached_result['status'] in ('ok', 'empty', 'partial') or time.time() - record['updated'] < 300:
                    result = cached_result
            # Parallel independent calls, with a single short-lived OCR subprocess.
            with ThreadPoolExecutor(max_workers=2) as pool:
                ocr_future = pool.submit(self.ocr.extract, decoded) if result is None else None
                vision_future = pool.submit(self.ai.describe_image, image, mime) if vision is None else None
                if ocr_future:
                    try:
                        result = ocr_future.result()
                    except (RuntimeError, OSError, ValueError):
                        result = {'status': 'unavailable', 'lines': [], 'notes': ['OCR 本次未完成']}
                    if path.stem.endswith(('_t', '_b')):
                        result.setdefault('notes', []).append('OCR依据缩略图，小字可能缺失')
                    if sticker and path.suffix == '.jpg':
                        result.setdefault('notes', []).append('OCR依据发送时已选中的表情预览')
                    result['vision_hash'] = vision_hash
                    with self.state:
                        self.state.execute('INSERT OR REPLACE INTO image_ocr_memory VALUES(?,?,?,?,?,?,?)',
                            (group_id, shard, ident, source_hash, self.ocr.revision,
                             json.dumps(result, ensure_ascii=False), int(time.time())))
                if vision_future:
                    try:
                        vision, _ = vision_future.result()
                        if sticker:
                            vision = ('（依据发送时已选中的表情预览，仅静态画面）' if path.suffix == '.jpg'
                                      else '（仅表情静态画面）') + vision
                        if path.stem.endswith(('_t', '_b')):
                            vision = '（依据缩略图，细节可能不清晰）' + vision
                        with self.state:
                            self.state.execute('INSERT OR REPLACE INTO image_memory VALUES(?,?,?,?,?)',
                                (group_id, shard, ident, vision_hash, vision[:1800]))
                    except (RuntimeError, OSError, ValueError):
                        if not result.get('lines'):
                            raise ImageUnavailable('视觉识别暂未成功，OCR也未取得可用文字') from None
            return self.combine(vision, result, question, max_chars)
        if old_vision or record:
            result = record['result'] if record and record['revision'] == self.ocr.revision else {
                'status': 'unavailable', 'lines': [], 'notes': ['本机原图不可用，仅有旧视觉摘要，尚未补做OCR']}
            vision = old_vision[1] if old_vision else None
            if result.get('vision_hash') and (not old_vision or old_vision[0] != result['vision_hash']):
                vision = None
            return self.combine(vision, result, question, max_chars)
        raise last_error
