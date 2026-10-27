"""Local group-scoped image decoding and cached vision descriptions.

V1/V2 format: 15-byte header, PKCS7 AES-ECB region, raw region, XOR tail.
Never download URLs from message XML or search another conversation's files.
"""
import hashlib
import io
import json
import re
import struct
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image


MAX_BYTES = 12 * 1024 * 1024
ImageCandidate = tuple[Path, bytes, bytes, str]


class ImageUnavailable(RuntimeError):
    pass


def is_image(data):
    return (data.startswith((b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', b'GIF8', b'wxgf'))
            or (data.startswith(b'RIFF') and data[8:12] == b'WEBP'))


def decode_dat(data, key=None, xor_key=None):
    if not data or len(data) > MAX_BYTES:
        raise ImageUnavailable('图片为空或超过大小限制')
    if is_image(data):
        return data
    if data[:6] in (b'\x07\x08V1\x08\x07', b'\x07\x08V2\x08\x07'):
        if data[:6] == b'\x07\x08V1\x08\x07':
            key = b'cfcd208495d565ef'
        if key is None or len(key) != 16 or len(data) < 31:
            raise ImageUnavailable('本机图片解密密钥尚未就绪')
        aes_size, xor_size = struct.unpack_from('<II', data, 6)
        encrypted_size = (aes_size // 16 + 1) * 16
        end = 15 + encrypted_size
        if end > len(data) or xor_size > len(data) - end:
            raise ImageUnavailable('图片文件不完整')
        try:
            decoder = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
            padded = decoder.update(data[15:end]) + decoder.finalize()
            unpad = padding.PKCS7(128).unpadder()
            head = unpad.update(padded) + unpad.finalize()
        except ValueError:
            raise ImageUnavailable('图片解密校验失败') from None
        if len(head) != aes_size or not is_image(head):
            raise ImageUnavailable('图片密钥与当前文件不匹配')
        raw_end = len(data) - xor_size
        if xor_size and xor_key is None:
            if head.startswith(b'\xff\xd8\xff') and xor_size >= 2:
                candidate = data[-2] ^ 0xff
                if data[-1] ^ candidate == 0xd9:
                    xor_key = candidate
            if xor_key is None:
                raise ImageUnavailable('图片尾部解密密钥缺失')
        tail = bytes(value ^ xor_key for value in data[raw_end:]) if xor_size else b''
        return head + data[end:raw_end] + tail
    for magic in (b'\xff\xd8\xff', b'\x89PNG', b'GIF8', b'RIFF'):
        k = data[0] ^ magic[0]
        if bytes(b ^ k for b in data[:len(magic)]) == magic:
            return bytes(b ^ k for b in data)
    raise ImageUnavailable('无法识别本机图片格式')


def normalize_image(data):
    """Decode fully, strip metadata and bound the upload size/resolution."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.width * im.height > 25_000_000:
                    raise ImageUnavailable('图片像素数超过限制')
                im.load()
                im.thumbnail((1600, 1600))
                output = io.BytesIO()
                im.convert('RGB').save(output, format='JPEG', quality=88)
                return output.getvalue(), 'image/jpeg'
    except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageUnavailable('图片尚不完整或格式暂不支持') from None


class ImageContext:
    def __init__(self, root: Path, state: Any, ai: Any, bot_id: str) -> None:
        """Initialize group-scoped image caches and optional UI fetching.

        Args:
            root: Bot workspace containing WeChat data and private configuration.
            state: Writable bot-state SQLite connection.
            ai: Client that describes normalized images.
            bot_id: Logged-in WeChat account ID.
        """
        self.root, self.state, self.ai, self.bot_id = Path(root), state, ai, bot_id
        self.image_fetcher: Callable[[str, Mapping[str, Any], bytes], None] | None = None
        self.image_closer: Callable[[str], None] | None = None
        state.execute('''CREATE TABLE IF NOT EXISTS image_memory(
            group_id TEXT, shard TEXT, local_id INTEGER, image_hash TEXT, summary TEXT,
            PRIMARY KEY(group_id,shard,local_id))''')
        state.execute('''CREATE TABLE IF NOT EXISTS image_fetch_attempts(
            group_id TEXT, shard TEXT, local_id INTEGER, preview_hash TEXT, attempted INTEGER,
            PRIMARY KEY(group_id,shard,local_id))''')
        state.commit()

    def cached(self, group_id: str, shard: str, local_id: int) -> str | None:
        """Return a previously stored description for one group image.

        Args:
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            local_id: Local image message ID.

        Returns:
            The cached image description, or None if this image has not been read.
        """
        row = self.state.execute('SELECT summary FROM image_memory WHERE group_id=? AND shard=? AND local_id=?',
                                 (group_id, shard, local_id)).fetchone()
        return row[0] if row else None

    def keys(self, account: Path, sample: bytes) -> tuple[bytes, int | None]:
        """Find and verify this account's local image decryption key.

        Args:
            account: Logged-in WeChat account directory.
            sample: Encrypted image attachment used to validate a candidate key.

        Returns:
            The verified AES key and optional XOR key.

        Raises:
            ImageUnavailable: No key can decode the supplied attachment.
        """
        key_path = self.root / 'image_keys.json'
        if key_path.exists():
            cfg = json.loads(key_path.read_text())
            if cfg.get('account') == account.name:
                try:
                    key = cfg['aes_key'].encode('ascii')
                    decode_dat(sample, key, cfg.get('xor_key'))
                    return key, cfg.get('xor_key')
                except (ValueError, KeyError, ImageUnavailable):
                    pass
        # Linux client versions using kvcomm-derived account image keys.
        # Verify each candidate against this file; never assume derivation works.
        kvcomm = self.root / 'data/wechat/.xwechat/net/kvcomm'
        for path in kvcomm.glob('*.statistic'):
            for value in path.name.removeprefix('key_').split('_')[:2]:
                if not value.isdigit():
                    continue
                for wxid in (self.bot_id, account.name):
                    key = hashlib.md5((value + wxid).encode()).hexdigest()[:16].encode()
                    xor = int(value) & 255
                    try:
                        decode_dat(sample, key, xor)
                        return key, xor
                    except ImageUnavailable:
                        continue
        raise ImageUnavailable('当前账号的图片解密密钥尚未就绪')

    def paths(
        self, account: Path, group_id: str, item: Mapping[str, Any], *,
        include_previews: bool = True,
    ) -> list[Path]:
        """Return this image's local attachment variants, highest quality first.

        Args:
            account: Logged-in WeChat account directory.
            group_id: Numeric ID of the current group.
            item: Message row containing its timestamp and attachment metadata.
            include_previews: Whether to include bubble and thumbnail caches.

        Returns:
            Existing image paths confined to this account's directory.
        """
        if not re.fullmatch(r'[0-9]+@chatroom', group_id):
            raise ImageUnavailable('无效图片会话')
        packed = item.get('packed_info_data') or b''
        if isinstance(packed, str):
            packed = packed.encode()
        hashes = list(dict.fromkeys(m.decode() for m in re.findall(rb'[0-9a-f]{32}', packed)))
        group_hash = hashlib.md5(group_id.encode()).hexdigest()
        month = datetime.fromtimestamp(item['create_time']).strftime('%Y-%m')
        folder = account / 'msg/attach' / group_hash / month / 'Img'
        bubble = account / 'cache' / month / 'Message' / group_hash / 'Bubble'
        paths = []
        for file_hash in hashes:
            variants = [folder / (file_hash + '.dat'), folder / (file_hash + '_h.dat')]
            if include_previews:
                variants.extend((bubble / (file_hash + '_b.dat'), folder / (file_hash + '_t.dat')))
            for path in variants:
                if path.is_file() and path.resolve().is_relative_to(account.resolve()):
                    paths.append(path)
        return paths

    def load_path(self, account: Path, path: Path) -> tuple[bytes, bytes, str]:
        """Decode and normalize one local image variant.

        Args:
            account: Logged-in WeChat account directory.
            path: A candidate path returned by ``paths``.

        Returns:
            Decoded source bytes, normalized upload bytes, and MIME type.
        """
        with path.open('rb') as f:
            data = f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ImageUnavailable('图片超过大小限制')
        key, xor = self.keys(account, data) if data[:6] == b'\x07\x08V2\x08\x07' else (None, None)
        decoded = decode_dat(data, key, xor)
        image, mime = normalize_image(decoded)
        return decoded, image, mime

    def load_first(
        self, account: Path, paths: list[Path],
    ) -> tuple[ImageCandidate | None, ImageUnavailable]:
        """Load the first decodable candidate while retaining the last local error.

        Args:
            account: Logged-in WeChat account directory.
            paths: Candidate attachment paths in quality order.

        Returns:
            The first decoded candidate and the most recent decode error.
        """
        last_error = ImageUnavailable('这张图片还没有下载到本机')
        for path in paths:
            try:
                decoded, image, mime = self.load_path(account, path)
            except (ImageUnavailable, OSError) as exc:
                last_error = (
                    exc if isinstance(exc, ImageUnavailable)
                    else ImageUnavailable('本机图片暂时无法读取')
                )
                continue
            return (path, decoded, image, mime), last_error
        return None, last_error

    def fetch_attempted_recently(
        self, group_id: str, shard: str, local_id: int, preview_hash: str,
    ) -> bool:
        """Check whether this unchanged preview recently triggered a UI fetch.

        Args:
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            local_id: Local image message ID.
            preview_hash: Hash of the normalized preview used for matching.

        Returns:
            True when the same preview was attempted within ten minutes.
        """
        row = self.state.execute(
            'SELECT preview_hash,attempted FROM image_fetch_attempts '
            'WHERE group_id=? AND shard=? AND local_id=?',
            (group_id, shard, local_id),
        ).fetchone()
        return bool(row and row[0] == preview_hash and time.time() - row[1] < 600)

    def remember_fetch_attempt(
        self, group_id: str, shard: str, local_id: int, preview_hash: str,
    ) -> None:
        """Record one bounded attempt to open this image in WeChat.

        Args:
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            local_id: Local image message ID.
            preview_hash: Hash of the normalized preview used for matching.
        """
        attempted = int(time.time())
        with self.state:
            self.state.execute(
                'DELETE FROM image_fetch_attempts WHERE attempted<?',
                (attempted - 600,),
            )
            self.state.execute(
                'INSERT OR REPLACE INTO image_fetch_attempts VALUES(?,?,?,?,?)',
                (group_id, shard, local_id, preview_hash, attempted),
            )
            self.state.execute(
                'DELETE FROM image_fetch_attempts WHERE rowid IN ( '
                'SELECT rowid FROM image_fetch_attempts ORDER BY attempted DESC '
                'LIMIT -1 OFFSET 1000)'
            )

    def forget_fetch_attempt(self, group_id: str, shard: str, local_id: int) -> None:
        """Clear the fetch cooldown after a full-resolution file is available.

        Args:
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            local_id: Local image message ID.
        """
        with self.state:
            self.state.execute(
                'DELETE FROM image_fetch_attempts WHERE group_id=? AND shard=? AND local_id=?',
                (group_id, shard, local_id),
            )

    def resolve_image(
        self, account: Path, group_id: str, shard: str, item: Mapping[str, Any],
    ) -> ImageCandidate:
        """Prefer a full-resolution attachment, requesting it from WeChat once if needed.

        Args:
            account: Logged-in WeChat account directory.
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            item: Image message row.

        Returns:
            A decoded full-resolution image, or a disclosed preview fallback.

        Raises:
            ImageUnavailable: No usable local image variant exists.
        """
        full_paths = self.paths(account, group_id, item, include_previews=False)
        full, _ = self.load_first(account, full_paths)
        if full:
            self.forget_fetch_attempt(group_id, shard, item['local_id'])
            return full

        all_paths = self.paths(account, group_id, item)
        preview_paths = [path for path in all_paths if path.stem.endswith(('_b', '_t'))]
        preview, preview_error = self.load_first(account, preview_paths)
        if preview is None:
            raise preview_error
        _, _, preview_image, _ = preview
        preview_hash = hashlib.sha256(preview_image).hexdigest()
        should_fetch = (
            self.image_fetcher is not None
            and not self.fetch_attempted_recently(group_id, shard, item['local_id'], preview_hash)
        )
        opened = False
        if should_fetch:
            self.remember_fetch_attempt(group_id, shard, item['local_id'], preview_hash)
            try:
                self.image_fetcher(group_id, item, preview_image)
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                opened = True
            try:
                if opened:
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        full_paths = self.paths(account, group_id, item, include_previews=False)
                        full, _ = self.load_first(account, full_paths)
                        if full:
                            self.forget_fetch_attempt(group_id, shard, item['local_id'])
                            return full
                        time.sleep(0.4)
            finally:
                if opened and self.image_closer:
                    try:
                        self.image_closer(group_id)
                    except (OSError, RuntimeError, ValueError):
                        pass
        if opened:
            full_paths = self.paths(account, group_id, item, include_previews=False)
            full, _ = self.load_first(account, full_paths)
            if full:
                self.forget_fetch_attempt(group_id, shard, item['local_id'])
                return full
        return preview

    def describe(
        self, account: Path, group_id: str, shard: str, item: Mapping[str, Any],
    ) -> str:
        """Describe one image, requesting a full-resolution local file first.

        Args:
            account: Logged-in WeChat account directory.
            group_id: Group containing the image.
            shard: Message database shard containing the image.
            item: Image message row.

        Returns:
            A cached or newly generated Chinese image description.

        Raises:
            ImageUnavailable: No usable image variant could be read.
        """
        cached = self.cached(group_id, shard, item['local_id'])
        if cached and '依据缩略图' not in cached and '依据微信预览图' not in cached:
            return cached
        try:
            path, _, image, mime = self.resolve_image(account, group_id, shard, item)
        except ImageUnavailable:
            if cached:
                return cached
            raise
        if cached and path.stem.endswith(('_b', '_t')):
            return cached
        summary, _ = self.ai.describe_image(image, mime)
        if path.stem.endswith(('_b', '_t')):
            summary = '（依据缩略图，原图未能读取）' + summary
        with self.state:
            self.state.execute('INSERT OR REPLACE INTO image_memory VALUES(?,?,?,?,?)',
                (group_id, shard, item['local_id'], hashlib.sha256(image).hexdigest(), summary))
        return summary
