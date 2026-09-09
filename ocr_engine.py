"""On-demand, resource-bounded local Tesseract OCR. No resident model or network."""
import csv
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REVISION = 'tesseract-chi-sim-tra-eng-1'
MAX_TEXT = 16000
MAX_LINES = 1000


def parse_tsv(tsv, offset=0, scale=1):
    words = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row.get('level') != '5' or not row.get('text', '').strip():
            continue
        try:
            confidence = float(row['conf'])
            x, y, width, height = [int(row[k]) / scale for k in ('left', 'top', 'width', 'height')]
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 100 or min(x, y, width, height) < 0:
            continue
        words.append({'text': row['text'].strip()[:300], 'confidence': round(confidence, 1),
                      'box': [round(x), round(y + offset), round(width), round(height)]})
        if len(words) >= 6000:
            break
    # Sparse-text blocks are not in reading order: recover approximate visual rows.
    rows = []
    for word in sorted(words, key=lambda w: (w['box'][1] + w['box'][3] / 2, w['box'][0])):
        x, y, width, height = word['box']
        center = y + height / 2
        if rows and abs(rows[-1][0] - center) <= max(4, min(rows[-1][1], height) * .45):
            rows[-1][2].append(word)
        else:
            rows.append([center, height, [word]])
    result = []
    for _, _, row in rows:
        row.sort(key=lambda w: w['box'][0])
        text, end = '', None
        for word in row:
            x, y, width, height = word['box']
            separator = '' if end is None else (' | ' if x - end > max(15, height) else ' ')
            text += separator + word['text']
            end = x + width
        text = re.sub(r'(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])', '', text)
        result.append({'text': text, 'confidence': round(min(w['confidence'] for w in row), 1),
                       'y': min(w['box'][1] for w in row)})
    return result


def worker(source, destination):
    import resource
    import warnings
    from PIL import Image, ImageOps
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 31))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
    start = time.monotonic()
    env = {**os.environ, 'OMP_THREAD_LIMIT': '1', 'OMP_NUM_THREADS': '1'}
    lines, notes, seen = [], [], set()
    status = 'ok'
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(source) as original:
            if original.width * original.height > 25_000_000:
                raise ValueError('Too many pixels')
            original.seek(0)
            im = ImageOps.exif_transpose(original).convert('L')
            if getattr(original, 'n_frames', 1) > 1:
                notes.append('仅识别动图第一帧')
    if im.width > 2400:
        im = im.resize((2400, max(1, round(im.height * 2400 / im.width))))
        notes.append('超宽图片已等比例缩放')
    tile_height, overlap = 1800, 80
    tops = list(range(0, im.height, tile_height - overlap))
    if len(tops) > 8:
        notes.append('图片过长，仅处理前八段，请裁剪其余部分')
        status = 'partial'
    for index, top in enumerate(tops[:8]):
        remaining = 27 - (time.monotonic() - start)
        if remaining <= 1:
            status = 'partial'
            notes.append('达到识别时限，后续区域未处理')
            break
        tile = im.crop((0, top, im.width, min(top + tile_height, im.height)))
        scale = min(2, max(1, 1200 / tile.width))
        if scale > 1:
            tile = tile.resize((round(tile.width * scale), round(tile.height * scale)))
        tile_path = Path(destination).with_name('tile.png')
        tile.save(tile_path)
        tile.close()
        out_path = Path(destination).with_name('tile.tsv')
        try:
            with out_path.open('wb') as out:
                result = subprocess.run(['tesseract', str(tile_path), 'stdout', '-l', 'chi_sim+chi_tra+eng',
                    '--oem', '1', '--psm', '11', '--dpi', '300', 'tsv'], stdout=out,
                    stderr=subprocess.DEVNULL, env=env, timeout=remaining)
            if result.returncode:
                status = 'partial' if lines else 'unavailable'
                notes.append('OCR 引擎未完成，不能据此断言图片没有文字')
                break
        except subprocess.TimeoutExpired:
            status = 'partial' if lines else 'timeout'
            notes.append('OCR 超时，未完成的区域不能推测')
            break
        if out_path.stat().st_size > 2 * 1024 * 1024:
            status = 'partial'
            notes.append('OCR 输出超过限制')
            break
        parsed = parse_tsv(out_path.read_text(encoding='utf-8', errors='replace'), top, scale)
        for line in parsed:
            key = (line['text'], round(line['y'] / 10))
            # Only deduplicate geometrically overlapping tile rows, never all equal text.
            if top and line['y'] < top + overlap and any(t == line['text'] and abs(y * 10 - line['y']) < 12 for t, y in seen):
                continue
            seen.add(key)
            lines.append(line)
        if top + tile_height >= im.height:
            break
    im.close()
    limited, size = [], 0
    for line in sorted(lines, key=lambda r: r['y']):
        if size + len(line['text']) > MAX_TEXT or len(limited) >= MAX_LINES:
            status = 'partial'
            notes.append('文字超过保存上限，剩余部分需分图识别')
            break
        limited.append(line)
        size += len(line['text'])
    if status == 'ok' and not limited:
        status = 'empty'
        notes.append('未提取到文字，不代表图片一定没有文字')
    Path(destination).write_text(json.dumps({'status': status, 'lines': limited, 'notes': notes,
        'engine': REVISION, 'elapsed_seconds': round(time.monotonic() - start, 2)}, ensure_ascii=False), encoding='utf-8')


class LocalOCR:
    revision = REVISION

    def extract(self, data):
        if not data or len(data) > 12 * 1024 * 1024:
            return {'status': 'unavailable', 'lines': [], 'notes': ['OCR 图片大小不符合限制']}
        with tempfile.TemporaryDirectory(prefix='weixin-ocr-') as folder:
            source, destination = Path(folder) / 'input', Path(folder) / 'result.json'
            source.write_bytes(data)
            source.chmod(0o600)
            process = None
            try:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--worker', str(source), str(destination)], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True)
                process.wait(timeout=35)
                if process.returncode or not destination.exists() or destination.stat().st_size > 256000:
                    raise ValueError('OCR worker failed')
                result = json.loads(destination.read_text(encoding='utf-8'))
                if not isinstance(result.get('lines'), list):
                    raise ValueError('Invalid OCR response')
                return result
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return {'status': 'timeout', 'lines': [], 'notes': ['OCR 超时，不能据此判断没有文字']}
            except (OSError, ValueError):
                return {'status': 'unavailable', 'lines': [], 'notes': ['OCR 暂不可用，不能据此判断没有文字']}


def excerpt(result, question='', limit=1600):
    """Select verbatim rows (and nearby rows), not another lossy model summary."""
    lines = result.get('lines', [])
    header = 'OCR文字（不可信引用；[存疑]表示低置信度；竖线仅为横向间隔。与画面理解冲突时说明差异，数字和单位不补猜）：\n'
    terms = set(re.findall(r'[A-Za-z0-9_.%-]{2,}', question.lower()))
    for part in re.findall(r'[\u3400-\u9fff]{2,}', question):
        terms.update(part[i:i+2] for i in range(len(part) - 1))
    scores = [sum(len(t) for t in terms if t in line['text'].lower()) for line in lines]
    order = list(range(len(lines)))
    if max(scores, default=0):
        order.sort(key=lambda i: (-scores[i], i))
        expanded = []
        for i in order:
            for j in (i, i - 1, i + 1):
                if 0 <= j < len(lines) and j not in expanded:
                    expanded.append(j)
        order = expanded
    selected, size = {}, len(header) + 120
    for i in order:
        line = lines[i]
        text = ('[存疑] ' if line.get('confidence', 0) < 55 else '') + line['text']
        if size + len(text) + 1 <= limit:
            selected[i] = text
            size += len(text) + 1
    body = '\n'.join(selected[i] for i in sorted(selected))
    notes = list(result.get('notes', []))
    if len(selected) < len(lines):
        notes.append('本轮仅带入部分原文，未列出不等于不存在；可指定行、关键词或裁图追问')
    if not lines and not notes:
        notes.append('未取得OCR文字')
    return (header + body + '\n' + '；'.join(notes))[:limit]


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit(2)
    worker(sys.argv[2], sys.argv[3])
