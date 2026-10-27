"""Constrained X11 reply delivery to a manually verified, currently open group."""
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from Xlib import display, error
from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parent
os.environ['DISPLAY'] = ':1'


def xdo(*args):
    return subprocess.check_output(['xdotool', *map(str, args)], text=True, timeout=5).strip()


def main_window(d, expected_signature: dict[str, int | str] | None = None) -> Window:
    """Find the unique visible WeChat window, optionally by its saved title.

    Args:
        d: X11 display connection.
        expected_signature: Saved group title signature used to disambiguate windows.

    Returns:
        The visible WeChat main window.

    Raises:
        RuntimeError: The WeChat window is missing or ambiguous.
    """
    prop = d.screen().root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), 0)
    candidates = []
    title_atom = d.intern_atom('_NET_WM_NAME')
    utf8_atom = d.intern_atom('UTF8_STRING')
    for identifier in prop.value if prop else []:
        w = d.create_resource_object('window', int(identifier))
        if 'wechat' in str(w.get_wm_class()).lower() and w.get_attributes().map_state == 2:
            title = w.get_full_property(title_atom, utf8_atom)
            if title and title.value.decode('utf-8', errors='replace') == '图片和视频':
                continue
            g = w.get_geometry()
            if g.width >= 800 and g.height >= 600:
                candidates.append(w)
    if expected_signature:
        matches = [w for w in candidates if signature(w) == expected_signature]
        if matches:
            candidates = matches
    if len(candidates) != 1:
        raise RuntimeError('Exactly one logged-in WeChat window is required')
    return candidates[0]


def signature(w):
    g = w.get_geometry()
    # Title strip only. Notification badges and message contents are excluded.
    data = w.get_image(280, 26, g.width - 360, 38, 2, 0xffffffff)
    img = Image.frombytes('RGB', (g.width - 360, 38), data.data, 'raw', 'BGRX')
    return {'width': g.width, 'height': g.height,
            'title_sha256': hashlib.sha256(img.tobytes()).hexdigest()}


def verify(d, w, template):
    if signature(w) != template:
        raise RuntimeError('Group title or window geometry changed; delivery stopped')
    active = d.screen().root.get_full_property(d.intern_atom('_NET_ACTIVE_WINDOW'), 0)
    if not active or int(active.value[0]) != w.id:
        raise RuntimeError('WeChat is not the active window')


def window_image(w: Window) -> Image.Image:
    """Capture the current WeChat window as an RGB image.

    Args:
        w: Verified WeChat top-level window.

    Returns:
        An in-memory screenshot of the window.
    """
    geometry = w.get_geometry()
    shot = w.get_image(0, 0, geometry.width, geometry.height, 2, 0xffffffff)
    return Image.frombytes('RGB', (geometry.width, geometry.height), shot.data, 'raw', 'BGRX')


def find_image_region(
    screen: Image.Image, template: Image.Image,
) -> tuple[int, int, int, int] | None:
    """Locate a recent image bubble by matching its cached preview in the group pane.

    Args:
        screen: Screenshot of the current group window.
        template: Decoded bubble preview for the target message.

    Returns:
        Window-relative image bounds, or None if the preview is not visible.
    """
    template = template.convert('RGB')
    if min(template.size) < 48:
        return None
    left, top = min(320, max(180, screen.width // 3)), 70
    right, bottom = screen.width - 15, screen.height - 110
    if right <= left or bottom <= top:
        return None
    pane = screen.crop((left, top, right, bottom))
    feature_size = (20, 20)
    target = template.resize(feature_size, Image.Resampling.LANCZOS)
    best_score, best_bounds = 256.0, None
    for scale in (0.4, 0.55, 0.7, 0.85, 1.0, 1.2):
        width = round(template.width * scale)
        height = round(template.height * scale)
        if width < 48 or height < 48 or width > pane.width or height > pane.height:
            continue
        step = max(5, min(width, height) // 30)
        for y in range(0, pane.height - height + 1, step):
            for x in range(0, pane.width - width + 1, step):
                patch = pane.crop((x, y, x + width, y + height)).resize(
                    feature_size, Image.Resampling.BILINEAR,
                )
                score = sum(ImageStat.Stat(ImageChops.difference(target, patch)).mean) / 3
                lower_tie = bool(
                    score == best_score and best_bounds and y > best_bounds[1]
                )
                if score < best_score or lower_tie:
                    best_score = score
                    best_bounds = (left + x, top + y, width, height)
    return best_bounds if best_score <= 32 else None


def open_image_preview(d, w: Window, template: str, preview: str) -> None:
    """Open a uniquely matched image in its verified group so WeChat can fetch it.

    Args:
        d: X11 display connection.
        w: Verified WeChat window.
        template: Saved title signature for the intended group.
        preview: Base64-encoded normalized image preview.

    Raises:
        ValueError: The preview is malformed or exceeds the size limit.
        RuntimeError: The exact preview cannot be safely found onscreen.
    """
    if not isinstance(preview, str) or len(preview) > 6 * 1024 * 1024:
        raise ValueError('Image preview exceeds the UI matching limit')
    try:
        raw = base64.b64decode(preview, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 4_000_000:
                raise ValueError('Image preview exceeds the pixel limit')
            source.load()
            target = source.convert('RGB')
    except (OSError, ValueError) as exc:
        raise ValueError('Image preview could not be decoded') from exc

    geometry = w.get_geometry()
    scroll_x, scroll_y = geometry.width - 60, geometry.height // 2
    xdo('key', '--clearmodifiers', 'Escape')
    xdo('mousemove', '--window', w.id, scroll_x, scroll_y)
    xdo('click', '--repeat', '16', '--delay', '12', '5')
    time.sleep(.3)
    match = None
    clicked = False
    try:
        for page in range(6):
            verify(d, w, template)
            match = find_image_region(window_image(w), target)
            if match:
                break
            if page < 5:
                xdo('click', '--repeat', '4', '--delay', '20', '4')
                time.sleep(.2)
        if not match:
            raise RuntimeError('Target image preview is not visible in the verified group')
        verify(d, w, template)
        x, y, width, height = match
        xdo('mousemove', '--window', w.id, x + width // 2, y + height // 2, 'click', '1')
        clicked = True
    finally:
        if not clicked and signature(w) == template:
            xdo('mousemove', '--window', w.id, scroll_x, scroll_y)
            xdo('click', '--repeat', '16', '--delay', '12', '5')


def save_profile(path, profiles, group_id, group_name, value):
    for other_id, profile in profiles.items():
        if other_id != group_id and profile.get('signature') == value:
            raise RuntimeError('Selected screen matches another group; profile update refused')
    os.umask(0o077)
    profiles[group_id] = {'group_name': group_name, 'signature': value}
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(profiles, ensure_ascii=False))
    temp.chmod(0o600)
    temp.replace(path)


def paste(text):
    proc = subprocess.Popen(['xclip', '-selection', 'clipboard', '-in', '-quiet'],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.stdin.write(text.encode())
        proc.stdin.close()
        time.sleep(.1)
        xdo('key', '--clearmodifiers', 'ctrl+v')
        time.sleep(.2)
        xdo('key', '--clearmodifiers', 'ctrl+a', 'ctrl+c')
        time.sleep(.1)
        copied = subprocess.check_output(['xclip', '-selection', 'clipboard', '-o'], timeout=3).decode()
        if copied != text:
            raise RuntimeError('Input text verification failed')
        xdo('key', '--clearmodifiers', 'Right')
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=3)


def select_group(w, group_name, unique_result=False):
    xdo('key', '--clearmodifiers', 'Escape')
    xdo('mousemove', '--window', w.id, 145, 42, 'click', '1')
    xdo('key', '--clearmodifiers', 'ctrl+a')
    paste(group_name)
    # Network suggestions vary in count. Locate the actual search popup and
    # choose its local-result row from the bottom, never a fixed screen row.
    d = display.Display(':1')
    root = d.screen().root
    origin = root.translate_coords(w.id, 0, 0)
    deadline = time.monotonic() + 5
    previous, stable_since, popup, popup_shot = None, time.monotonic(), None, None
    while time.monotonic() < deadline:
        popups = []
        for child in root.query_tree().children:
            g = child.get_geometry()
            if (child.get_attributes().map_state == 2 and 300 <= g.width <= 500
                    and 120 <= g.height <= 600 and abs(g.x - origin.x - 74) <= 4
                    and abs(g.y - origin.y - 58) <= 4):
                popups.append(g)
        shape = None if len(popups) != 1 else (popups[0].x, popups[0].y, popups[0].width, popups[0].height)
        shot = None if shape is None else root.get_image(*shape, 2, 0xffffffff)
        state = None if shot is None else (shape, hashlib.sha256(shot.data).digest())
        if state != previous:
            previous, stable_since = state, time.monotonic()
        if state and time.monotonic() - stable_since >= .8:
            popup = popups[0]
            popup_shot = shot
            break
        time.sleep(.1)
    if popup is None:
        raise RuntimeError('Local group search popup could not be located')
    # The popup can reorder network suggestions around local results. Its first
    # full-size avatar is the exact group result; matching chat-history avatars
    # follow it. Magnifying-glass network rows have no full-size avatar.
    image = Image.frombytes('RGB', (popup.width, popup.height), popup_shot.data, 'raw', 'BGRX')
    avatar_rows = []
    for y in range(popup.height):
        colored = sum(1 for x in range(8, min(48, popup.width))
                      if sum(255 - channel for channel in image.getpixel((x, y))) > 30)
        if colored >= 18:
            avatar_rows.append(y)
    regions = []
    for y in avatar_rows:
        if not regions or y > regions[-1][1] + 3:
            regions.append([y, y])
        else:
            regions[-1][1] = y
    avatars = [region for region in regions if region[0] > 30 and region[1] - region[0] >= 24]
    if not avatars:
        raise RuntimeError('Exact local group result could not be identified')
    # WeChat keyboard-selects the exact local group result. Enter is stable even
    # when asynchronous network suggestions reorder the popup around it.
    if unique_result:
        # Unnamed/member-derived groups are not keyboard-selected by WeChat.
        # Require one local avatar result and an unchanged popup before clicking.
        if len(avatars) != 1:
            raise RuntimeError('Unnamed group search is ambiguous')
        current = root.get_image(popup.x, popup.y, popup.width, popup.height, 2, 0xffffffff)
        if current.data != popup_shot.data:
            raise RuntimeError('Search results changed before selection')
        # Clicking the avatar toggles a preview instead of opening the chat.
        # Double-click the label, away from both avatar and info controls.
        xdo('mousemove', popup.x + 124, popup.y + sum(avatars[0]) // 2,
            'click', '--repeat', 2, '--delay', 120, '1')
    else:
        xdo('key', '--clearmodifiers', 'Return')
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        visible = False
        for child in root.query_tree().children:
            try:
                current = child.get_geometry()
                if (child.get_attributes().map_state == 2 and current.x == popup.x
                        and current.y == popup.y and current.width == popup.width
                        and current.height == popup.height):
                    visible = True
                    break
            except error.XError:
                continue
        if not visible:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Exact group result did not open')
    time.sleep(.4)


def main() -> None:
    """Run one constrained, verified WeChat UI action."""
    action = sys.argv[1]
    payload = {} if action == 'logged-in' else json.load(sys.stdin)
    if action != 'logged-in':
        group_id = payload.get('group_id', '')
        group_name = payload.get('group_name', '')
        if not re.fullmatch(r'[0-9]+@chatroom', group_id) or not group_name.strip():
            raise ValueError('Explicit group ID and name are required')
    profiles_path = ROOT / 'titles.json'
    profile_hint = None
    if action == 'close-image':
        profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
        profile_hint = profiles.get(group_id)
        if not profile_hint or profile_hint.get('group_name') != group_name:
            raise RuntimeError('Unverified target group')
    d = display.Display(':1')
    w = main_window(d, profile_hint.get('signature') if profile_hint else None)
    if action == 'logged-in':
        print('WeChat is logged in')
        return
    if action != 'close-image':
        xdo('windowactivate', '--sync', w.id)
        time.sleep(.1)
    if action == 'capture-title':
        profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
        save_profile(profiles_path, profiles, group_id, group_name, signature(w))
        print('Title verification template saved')
        return
    if action == 'open':
        select_group(w, group_name, payload.get('select_unique_result', False))
        print('Group candidate opened; visual verification required')
        return
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    profile = profiles.get(group_id)
    if action == 'ready':
        for attempt in range(2):
            if profile and profile['group_name'] == group_name and signature(w) == profile['signature']:
                break
            select_group(w, group_name, payload.get('select_unique_result', False))
            deadline = time.monotonic() + 1.5
            previous, stable_since = None, time.monotonic()
            while time.monotonic() < deadline:
                current = signature(w)
                if current != previous:
                    previous, stable_since = current, time.monotonic()
                if time.monotonic() - stable_since >= .3:
                    break
                time.sleep(.1)
            if profile and profile['group_name'] == group_name and current == profile['signature']:
                break
            if payload.get('allow_profile_refresh'):
                save_profile(profiles_path, profiles, group_id, group_name, current)
                profile = profiles[group_id]
                break
        if not profile or profile['group_name'] != group_name:
            raise RuntimeError('Unverified target group')
        verify(d, w, profile['signature'])
        print('Group title verified')
        return
    if not profile or profile['group_name'] != group_name:
        raise RuntimeError('Unverified target group')
    template = profile['signature']
    if action == 'close-image':
        if signature(w) != template:
            raise RuntimeError('Group title changed; image preview was left untouched')
    else:
        verify(d, w, template)
    if action == 'open-image':
        if type(payload.get('local_id')) is not int or payload['local_id'] <= 0:
            raise ValueError('Invalid image message ID')
        open_image_preview(d, w, template, payload.get('preview', ''))
        print('Image preview opened in verified group')
        return
    if action == 'close-image':
        if signature(w) != template:
            raise RuntimeError('Group title changed; image preview was left untouched')
        active = d.screen().root.get_full_property(d.intern_atom('_NET_ACTIVE_WINDOW'), 0)
        if not active:
            raise RuntimeError('No active window; image preview was left untouched')
        active_window = d.create_resource_object('window', int(active.value[0]))
        if 'wechat' not in str(active_window.get_wm_class()).lower():
            raise RuntimeError('WeChat preview is not active; image preview was left untouched')
        xdo('key', '--clearmodifiers', 'Escape')
        time.sleep(.2)
        verify(d, w, template)
        geometry = w.get_geometry()
        xdo('mousemove', '--window', w.id, geometry.width - 60, geometry.height // 2)
        xdo('click', '--repeat', '16', '--delay', '12', '5')
        print('Image preview closed in verified group')
        return
    if action == 'check':
        print('Group title verified')
        return
    if action not in ('draft-check', 'send'):
        raise ValueError('Unknown action')
    text = payload['text']
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError('Invalid reply text')
    g = w.get_geometry()
    xdo('mousemove', '--window', w.id, g.width // 2, g.height - 80, 'click', '1')
    xdo('key', '--clearmodifiers', 'ctrl+a')
    # Do not overwrite a human draft: selecting/copying an empty editor leaves
    # clipboard unchanged, so use a fresh sentinel before Ctrl+C.
    sentinel = '__WEIXIN_EMPTY_EDITOR__'
    keeper = subprocess.Popen(['xclip', '-selection', 'clipboard', '-in', '-quiet'],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        keeper.stdin.write(sentinel.encode()); keeper.stdin.close()
        time.sleep(.1)
        xdo('key', '--clearmodifiers', 'ctrl+c')
        time.sleep(.1)
        old = subprocess.check_output(['xclip', '-selection', 'clipboard', '-o'], timeout=3).decode()
        if old not in ('', sentinel):
            raise RuntimeError('Input already contains a draft; delivery stopped')
    finally:
        if keeper.poll() is None:
            keeper.terminate(); keeper.wait(timeout=3)
    paste(text)
    verify(d, w, template)
    if action == 'send':
        xdo('key', '--clearmodifiers', 'Return')
        print('Reply submitted')
    else:
        xdo('key', '--clearmodifiers', 'ctrl+a', 'BackSpace')
        print('Draft verified and cleared; no message sent')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
