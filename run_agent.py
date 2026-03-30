import os
import sys
import random
import time
import ctypes
import zlib
from ctypes import wintypes
from collections import defaultdict
import argparse
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Listener, Key

try:
    import mss
except ImportError:
    mss = None


start_agent = False
stop_agent = False
pause_agent = False
DEFAULT_WINDOW_TITLES = ["Dota 2"]
WINDOW_TITLE_EXCLUDES = ["visual studio code", "cursor", "pycharm", "notepad++", "sublime text"]
DEFAULT_MATCH_CONFIDENCE = 0.72
DEFAULT_ACTION_DELAY = 0.5
DEFAULT_LOOKAHEAD_DEPTH = 3
DEFAULT_SETTLE_TIMEOUT = 1.6
DEFAULT_PERF_SUMMARY_EVERY = 10
DEFAULT_IDLE_POLL_INTERVAL = 0.3
DEFAULT_SETTLE_POLL_INTERVAL = 0.035
DEFAULT_SETTLE_STABLE_FRAMES = 3
DEFAULT_SETTLE_DIFF_THRESHOLD = 2.4
DEFAULT_SETTLE_DOWNSCALE = 56
DEFAULT_AUTO_RECALIBRATE = True
DEFAULT_RECALIBRATE_INTERVAL = 30
DEFAULT_RECALIBRATE_PADDING = 14
DEFAULT_RECALIBRATE_STEP = 4
DEFAULT_RECALIBRATE_BAD_SCORE = 0.43
DEFAULT_RECALIBRATE_PERIODIC_SCORE = 0.395
DEFAULT_RECALIBRATE_INEFFECTIVE_STREAK = 2
DEFAULT_RECALIBRATE_COOLDOWN_CYCLES = 10
DEFAULT_RECOGNITION_RETRY_SAMPLES = 3
DEFAULT_RECOGNITION_RETRY_INTERVAL = 0.06
DEFAULT_CONFIDENCE_PAUSE_THRESHOLD = 0.50
DEFAULT_CONFIDENCE_PAUSE_STREAK = 2
DEFAULT_CONFIDENCE_RESUME_THRESHOLD = 0.44
DEFAULT_CONFIDENCE_RESUME_STREAK = 2
DEFAULT_FAILED_ACTION_COOLDOWN = 0.35
DEFAULT_DRAG_HOLD_BEFORE_MOVE = 0.020
DEFAULT_DRAG_HOLD_AFTER_MOVE = 0.045
DEFAULT_DRAG_HOLD_AFTER_MOVE_RETRY = 0.065
HSV_SAT_THRESHOLD = 35
HSV_VAL_THRESHOLD = 30
MORPH_KERNEL_SIZE = 3
MIN_CONTOUR_AREA = 8.0
HUE_HIST_BINS = 30
SAT_HIST_BINS = 32
HUE_WEIGHT = 0.35
SAT_WEIGHT = 0.30
SHAPE_WEIGHT = 0.15
EDGE_WEIGHT = 0.20
STANDARD_OBJECT_RATIO = 0.68
_MSS_CTX = None
_BOARD_TEMPLATE_CACHE = {}


class PerfTracker:
    def __init__(self, enabled=False, summary_every=10):
        self.enabled = bool(enabled)
        self.summary_every = max(1, int(summary_every))
        self._metrics = defaultdict(list)
        self._cycles = 0
        self._action_cycles = 0
        self._skipped_cycles = 0

    def add(self, name, seconds):
        if not self.enabled:
            return
        self._metrics[name].append(float(seconds) * 1000.0)

    def mark_cycle(self, action_taken=False, skipped=False):
        if not self.enabled:
            return
        self._cycles += 1
        if action_taken:
            self._action_cycles += 1
        if skipped:
            self._skipped_cycles += 1

        if self._cycles % self.summary_every == 0:
            self.print_summary()

    def print_summary(self):
        if not self.enabled:
            return
        print("\n[PERF] ===== cycle profile summary =====")
        print(f"[PERF] cycles={self._cycles}, action_cycles={self._action_cycles}, skipped={self._skipped_cycles}")
        for name in sorted(self._metrics.keys()):
            values = self._metrics[name]
            if not values:
                continue
            avg_ms = sum(values) / len(values)
            max_ms = max(values)
            print(f"[PERF] {name}: avg={avg_ms:.2f}ms max={max_ms:.2f}ms count={len(values)}")
        print("[PERF] =================================")


def on_press(key):
    global start_agent
    global stop_agent
    global pause_agent
    try:
        if key.char == 'b':
            start_agent = True
        elif key.char in ('p', 'q'):
            pause_agent = not pause_agent
    except AttributeError:
        if start_agent and key == Key.esc:
            stop_agent = True
            # return False


def get_image_array(image):
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


def _capture_with_mss(region=None):
    global _MSS_CTX
    if mss is None:
        return None
    if _MSS_CTX is None:
        _MSS_CTX = mss.mss()

    if region is None:
        monitor = _MSS_CTX.monitors[0]
    else:
        left, top, width, height = normalize_region(region)
        monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}

    frame = np.asarray(_MSS_CTX.grab(monitor), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    return frame[:, :, :3]


def capture_screen_array(region=None):
    region = normalize_region(region)
    frame = None
    try:
        frame = _capture_with_mss(region=region)
    except Exception:
        frame = None
    if frame is not None:
        return frame

    # pyautogui expects region=(left, top, width, height)
    screenshot = pyautogui.screenshot(region=region)
    return get_image_array(screenshot)


def get_window_rect_by_title():
    if os.name != "nt":
        return None

    user32 = ctypes.windll.user32
    matches = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True

        title_len = user32.GetWindowTextLengthW(hwnd)
        if title_len <= 0:
            return True

        title_buffer = ctypes.create_unicode_buffer(title_len + 1)
        user32.GetWindowTextW(hwnd, title_buffer, title_len + 1)
        title = title_buffer.value.strip()
        if not title:
            return True

        title_lower = title.lower()
        if any(excluded in title_lower for excluded in WINDOW_TITLE_EXCLUDES):
            return True
        if not any(keyword.lower() in title_lower for keyword in DEFAULT_WINDOW_TITLES):
            return True

        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return True

        top_left = wintypes.POINT(rect.left, rect.top)
        bottom_right = wintypes.POINT(rect.right, rect.bottom)
        user32.ClientToScreen(hwnd, ctypes.byref(top_left))
        user32.ClientToScreen(hwnd, ctypes.byref(bottom_right))

        width = bottom_right.x - top_left.x
        height = bottom_right.y - top_left.y
        if width < 200 or height < 200:
            return True

        area = width * height
        matches.append((area, title, (top_left.x, top_left.y, width, height)))
        return True

    user32.EnumWindows(enum_windows_proc, 0)
    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, title, rect = matches[0]
    print("Detected game window:", title, "region:", rect)
    return rect


def normalize_region(region):
    if region is None:
        return None
    if len(region) != 4:
        return None
    x, y, width, height = [int(v) for v in region]
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _match_template_multiscale(search_gray, search_edge, target_image, confidence=0.72, scale_min=0.55, scale_max=1.45, scale_steps=19):
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    best = None
    for scale in np.linspace(scale_min, scale_max, scale_steps):
        scaled_width = max(1, int(target_gray.shape[1] * scale))
        scaled_height = max(1, int(target_gray.shape[0] * scale))
        if scaled_width >= search_gray.shape[1] or scaled_height >= search_gray.shape[0]:
            continue

        resized_gray = cv2.resize(target_gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        resized_edge = cv2.Canny(resized_gray, 40, 120)

        for src, tmpl in ((search_gray, resized_gray), (search_edge, resized_edge)):
            result = cv2.matchTemplate(src, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best["score"]:
                best = {
                    "score": max_val,
                    "top_left": max_loc,
                    "bottom_right": (max_loc[0] + scaled_width, max_loc[1] + scaled_height),
                }

    if best is not None and best["score"] >= confidence:
        return best
    return None


def _load_board_templates(resource_dir):
    key = os.path.abspath(resource_dir)
    cached = _BOARD_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached

    top_left_template = cv2.imread(os.path.join(resource_dir, 'topLeft.jpg'), cv2.IMREAD_COLOR)
    bottom_right_template = cv2.imread(os.path.join(resource_dir, 'botRight.jpg'), cv2.IMREAD_COLOR)
    _BOARD_TEMPLATE_CACHE[key] = (top_left_template, bottom_right_template)
    return top_left_template, bottom_right_template


def load_elem_images(directory):
    elem_images = {}
    for filename in os.listdir(directory):
        if filename[0].isdigit():
            image_path = os.path.join(directory, filename)
            elem_name = filename.split('.')[0]
            elem_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if elem_image is not None:
                elem_images[elem_name] = elem_image
    return elem_images


def find_game_board(resource_dir, search_region=None, confidence=0.72):
    '''
    Set show_board=True to check the board image
    Make sure the image shown is similar to board_sample.pgn
    '''
    search_region = normalize_region(search_region)
    image_array = capture_screen_array(region=search_region)
    origin_x, origin_y = (search_region[0], search_region[1]) if search_region else (0, 0)

    top_left_template, bottom_right_template = _load_board_templates(resource_dir)
    if top_left_template is None or bottom_right_template is None:
        return None, None

    search_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    search_edge = cv2.Canny(search_gray, 40, 120)
    top_left_loc = _match_template_multiscale(search_gray, search_edge, top_left_template, confidence=confidence)
    bottom_right_loc = _match_template_multiscale(search_gray, search_edge, bottom_right_template, confidence=confidence)

    # Return None when board corner not found
    if top_left_loc is None or bottom_right_loc is None:
        return None, None

    get_middle = lambda x: (x["top_left"][0] + (x["bottom_right"][0] - x["top_left"][0]) // 2,
                            x["top_left"][1] + (x["bottom_right"][1] - x["top_left"][1]) // 2)
    top_left_local = get_middle(top_left_loc)
    bottom_right_local = get_middle(bottom_right_loc)
    top_left = (top_left_local[0] + origin_x, top_left_local[1] + origin_y)
    bottom_right = (bottom_right_local[0] + origin_x, bottom_right_local[1] + origin_y)

    x, y, w, h = top_left[0], top_left[1], bottom_right[0], bottom_right[1]

    # Return None when board corner location is incorrectly identified
    if w - x < 0 or h - y < 0:
        return None, None
    if (w - x) < 200 or (h - y) < 200:
        return None, None

    side_x, side_y = (w - x), (h - y)
    side_min = min(side_x, side_y)
    side_max = max(side_x, side_y)
    aspect_error = abs(side_x - side_y) / max(side_x, side_y)
    if aspect_error > 0.18:
        side = min(side_x, side_y)
        bottom_right = (x + side, y + side)
        side_min = side
        side_max = side

    # Filter obvious false positives: board is a large square in the window.
    if side_min < 300:
        return None, None
    if search_region is not None:
        window_w, window_h = search_region[2], search_region[3]
        if side_min < 0.3 * min(window_w, window_h):
            return None, None
        if side_max > 0.98 * max(window_w, window_h):
            return None, None

    print('board top left corner:', top_left)
    print('board bottom right corner:', bottom_right)
    print('corner score(top_left/bottom_right):', round(top_left_loc["score"], 3), round(bottom_right_loc["score"], 3))

    return top_left, bottom_right


def resolve_board_coordinates(resource_dir):
    # Identify the game board from the screen
    while True:
        search_region = get_window_rect_by_title()
        top_left, bottom_right = find_game_board(
            resource_dir,
            search_region=search_region,
            confidence=DEFAULT_MATCH_CONFIDENCE
        )
        if top_left is None or bottom_right is None:
            if search_region is None:
                print('Cannot locate Dota2 window, fallback to full screen board matching.')
            print('Cannot locate the game board, make sure the game is on the screen.')
            time.sleep(5)
        else:
            return top_left, bottom_right


def show_board_if_needed(show, top_left, bottom_right):
    if not show:
        return
    x, y = top_left
    w, h = bottom_right
    board = pyautogui.screenshot(region=(x, y, w - x, h - y))
    board.show()


def choose_action_with_repeat_guard(agent, actions, board_signature, last_board_signature, last_action, lookahead_depth):
    action = agent.choose_best_action(actions, lookahead_depth=lookahead_depth)
    action = agent.normalize_action(action)

    # Guard against repeating the same ineffective action on an unchanged board.
    if board_signature == last_board_signature and action == last_action and action is not None:
        alt_action = agent.choose_best_action(
            actions,
            lookahead_depth=lookahead_depth,
            banned_actions={action}
        )
        alt_action = agent.normalize_action(alt_action)
        if alt_action:
            return alt_action, False
        print("Board unchanged and no alternate valid action found; skip this cycle.")
        return None, True

    return action, False


def execute_action_cycle(agent, action, disable_wait_settle, settle_timeout, perf_tracker=None):
    t_take = time.perf_counter()
    action = agent.take_action(action)
    if perf_tracker is not None:
        perf_tracker.add("take_action", time.perf_counter() - t_take)

    settled = True
    if action and not disable_wait_settle:
        t_wait = time.perf_counter()
        settled = agent.wait_until_board_settled(timeout=settle_timeout)
        if perf_tracker is not None:
            perf_tracker.add("wait_board_settle", time.perf_counter() - t_wait)
        if not settled:
            print("Board settle wait timeout; retry swap with conservative gesture.")
            t_retry = time.perf_counter()
            retried = agent.retry_action(action)
            if perf_tracker is not None:
                perf_tracker.add("retry_take_action", time.perf_counter() - t_retry)

            if retried:
                t_wait_retry = time.perf_counter()
                settled = agent.wait_until_board_settled(timeout=max(0.4, settle_timeout * 0.75))
                if perf_tracker is not None:
                    perf_tracker.add("wait_board_settle_retry", time.perf_counter() - t_wait_retry)

            if not settled:
                print("Retry also failed to settle; continue with next cycle.")
                # Avoid back-to-back failed swaps when the board is still in transition.
                time.sleep(DEFAULT_FAILED_ACTION_COOLDOWN)
    return action, settled


def print_cycle_log(elem_array, elem_sub_array, action, confidence_score, elapsed_sec):
    def format_subclass(code):
        code = int(code)
        if code <= 0:
            return "0_0"
        return f"{code // 10}_{code % 10}"

    print(elem_array)
    print("subclass")
    formatted = np.vectorize(format_subclass)(elem_sub_array)
    print(formatted)
    print('action', action)
    print('confidence', round(float(confidence_score), 4))
    print('time cost', elapsed_sec)
    print('=' * 50)


def wait_for_start_signal():
    while not start_agent:
        time.sleep(1)
    pyautogui.click(interval=0.5)
    time.sleep(4)


def run_main_cycle(agent, wait_static, lookahead_depth, disable_wait_settle, settle_timeout,
                   last_board_signature, last_action, perf_tracker=None,
                   idle_poll_interval=DEFAULT_IDLE_POLL_INTERVAL):
    if stop_agent:
        print("Program stopped by user.")
        return True, last_board_signature, last_action

    if pause_agent:
        print("Program paused by user. Press 'p' or 'q' again to unpause.")
        time.sleep(max(0.05, float(idle_poll_interval)))
        return False, last_board_signature, last_action

    t_cycle = time.perf_counter()
    t1 = time.time()
    agent._cycle_index += 1
    avg_confidence_score = agent.refresh_board_state()
    if perf_tracker is not None:
        perf_tracker.add("refresh_board_state", time.perf_counter() - t_cycle)

    if agent.should_attempt_recalibrate(avg_confidence_score):
        t_recal = time.perf_counter()
        recalibrated_score = agent.try_recalibrate_region(avg_confidence_score)
        if perf_tracker is not None:
            perf_tracker.add("recalibrate_total", time.perf_counter() - t_recal)
        if recalibrated_score is not None:
            t_refresh = time.perf_counter()
            avg_confidence_score = agent.refresh_board_state()
            if perf_tracker is not None:
                perf_tracker.add("refresh_after_recalibrate", time.perf_counter() - t_refresh)

    if agent.should_pause_for_high_deviation(avg_confidence_score):
        print(
            f"Recognition deviation too high ({avg_confidence_score:.4f}), "
            "treat as paused and skip this cycle."
        )
        if perf_tracker is not None:
            perf_tracker.add("cycle_total", time.perf_counter() - t_cycle)
            perf_tracker.mark_cycle(action_taken=False, skipped=True)
        time.sleep(max(0.05, float(idle_poll_interval)))
        return False, last_board_signature, last_action

    should_act = agent.should_take_action(wait_static)
    if not should_act:
        if perf_tracker is not None:
            perf_tracker.add("cycle_total", time.perf_counter() - t_cycle)
            perf_tracker.mark_cycle(action_taken=False, skipped=True)
        time.sleep(max(0.05, float(idle_poll_interval)))
        return False, last_board_signature, last_action

    t_actions = time.perf_counter()
    actions = agent.get_action()
    if perf_tracker is not None:
        perf_tracker.add("get_action_candidates", time.perf_counter() - t_actions)

    board_signature = agent.get_board_signature()
    if last_action is not None and board_signature == last_board_signature:
        agent._ineffective_streak += 1
    else:
        agent._ineffective_streak = 0

    t_choose = time.perf_counter()
    action, should_skip = choose_action_with_repeat_guard(
        agent=agent,
        actions=actions,
        board_signature=board_signature,
        last_board_signature=last_board_signature,
        last_action=last_action,
        lookahead_depth=lookahead_depth
    )
    if perf_tracker is not None:
        perf_tracker.add("choose_best_action", time.perf_counter() - t_choose)

    if should_skip:
        # Force fresh recognition to avoid stale/repeated retries on unchanged signatures.
        t_resync = time.perf_counter()
        agent.force_resync_board_state()
        if perf_tracker is not None:
            perf_tracker.add("force_resync", time.perf_counter() - t_resync)
        time.sleep(max(0.05, float(idle_poll_interval)))
        if perf_tracker is not None:
            perf_tracker.add("cycle_total", time.perf_counter() - t_cycle)
            perf_tracker.mark_cycle(action_taken=False, skipped=True)
        return False, last_board_signature, last_action

    action, settled = execute_action_cycle(
        agent=agent,
        action=action,
        disable_wait_settle=disable_wait_settle,
        settle_timeout=settle_timeout,
        perf_tracker=perf_tracker
    )
    if action and not settled:
        # If action likely failed or board is still unstable, force multi-sample refresh.
        t_resync = time.perf_counter()
        agent.force_resync_board_state()
        if perf_tracker is not None:
            perf_tracker.add("force_resync", time.perf_counter() - t_resync)

    if perf_tracker is not None:
        perf_tracker.add("cycle_total", time.perf_counter() - t_cycle)
        perf_tracker.mark_cycle(action_taken=bool(action), skipped=False)

    print_cycle_log(
        elem_array=agent.elem_array,
        elem_sub_array=agent.elem_sub_array,
        action=action,
        confidence_score=avg_confidence_score,
        elapsed_sec=time.time() - t1
    )
    return False, board_signature, action


class ElementMatcher:
    def __init__(self, elem_images):
        self._elem_images = elem_images
        self._grid_size = None
        self._type_variant_images = defaultdict(list)
        self._base_template_contours = {}
        self._base_template_hists = {}
        self._base_template_sat_hists = {}
        self._base_template_edges = {}
        self._type_variant_features = {}
        self._build_base_templates_by_type()

    def _build_base_templates_by_type(self):
        for name, image in self._elem_images.items():
            try:
                piece_type = int(name.split("_")[0])
            except (ValueError, IndexError):
                continue
            if 1 <= piece_type <= 6:
                self._type_variant_images[piece_type].append((name, image))

    def _extract_piece_mask(self, image):
        h, w, _ = image.shape
        if h <= 2 or w <= 2:
            return None, None

        crop = image[h // 8: h - h // 8, w // 8: w - w // 8]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[:, :, 1] > HSV_SAT_THRESHOLD
        val_mask = hsv[:, :, 2] > HSV_VAL_THRESHOLD
        mask = np.logical_and(sat_mask, val_mask).astype(np.uint8) * 255

        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return crop, mask

    def _extract_grid_features_once(self, image):
        crop, mask = self._extract_piece_mask(image)
        if crop is None or mask is None:
            return None, None, None, None

        non_zero = cv2.countNonZero(mask)
        if non_zero < 10:
            return None, None, None, None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([hsv], [0], mask, [HUE_HIST_BINS], [0, 180])
        if hist is not None:
            cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

        sat_hist = cv2.calcHist([hsv], [1], mask, [SAT_HIST_BINS], [0, 256])
        if sat_hist is not None:
            cv2.normalize(sat_hist, sat_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = None
        if contours:
            cand = max(contours, key=cv2.contourArea)
            if float(cv2.contourArea(cand)) >= MIN_CONTOUR_AREA:
                contour = cand

        gray = cv2.bitwise_and(gray, gray, mask=mask)
        edge = cv2.Canny(gray, 50, 140)
        return hist, sat_hist, contour, edge

    def _standardize_by_contour(self, image, target_width, target_height):
        """
        Normalize piece scale/position using the largest contour:
        extract object bbox -> resize to a fixed ratio -> center on black canvas.
        """
        crop, mask = self._extract_piece_mask(image)
        if crop is None or mask is None:
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

        x, y, w, h = cv2.boundingRect(contour)
        obj = crop[y:y + h, x:x + w]
        obj_mask = mask[y:y + h, x:x + w]
        obj_fg = cv2.bitwise_and(obj, obj, mask=obj_mask)

        max_side = max(1, int(min(target_width, target_height) * STANDARD_OBJECT_RATIO))
        scale = max_side / max(1, max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        obj_fg = cv2.resize(obj_fg, (new_w, new_h), interpolation=interp)
        obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        offset_x = (target_width - new_w) // 2
        offset_y = (target_height - new_h) // 2
        roi = canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
        roi[obj_mask > 0] = obj_fg[obj_mask > 0]
        return canvas

    def _extract_main_contour(self, image):
        _, _, contour, _ = self._extract_grid_features_once(image)
        return contour

    def _extract_hue_hist(self, image):
        hist, _, _, _ = self._extract_grid_features_once(image)
        return hist

    def _extract_sat_hist(self, image):
        _, sat_hist, _, _ = self._extract_grid_features_once(image)
        return sat_hist

    def _extract_edge_map(self, image):
        _, _, _, edge = self._extract_grid_features_once(image)
        return edge

    def set_grid_size(self, width, height):
        self._grid_size = (width, height)
        self._base_template_contours = {}
        self._base_template_hists = {}
        self._base_template_sat_hists = {}
        self._base_template_edges = {}
        self._type_variant_features = {}

        for piece_type in range(1, 7):
            variants = self._type_variant_images.get(piece_type, [])
            if not variants:
                continue

            # Contour template prefers x_1 for shape stability.
            contour_template = None
            for name, img in variants:
                if name.endswith("_1"):
                    contour_template = img
                    break
            if contour_template is None:
                contour_template = variants[0][1]

            contour_image = cv2.resize(contour_template, (width, height), interpolation=cv2.INTER_AREA)
            contour = self._extract_main_contour(contour_image)
            if contour is not None:
                self._base_template_contours[piece_type] = contour
            edge = self._extract_edge_map(contour_image)
            if edge is not None:
                self._base_template_edges[piece_type] = edge

            # Hue template uses all variants for robustness.
            hist_list = []
            sat_hist_list = []
            variant_features = []
            for _, img in variants:
                variant_name = _.strip()
                try:
                    variant_idx = int(variant_name.split("_")[1])
                except (ValueError, IndexError):
                    variant_idx = 1
                variant_code = piece_type * 10 + variant_idx

                resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                standardized = self._standardize_by_contour(resized, width, height)
                hist = self._extract_hue_hist(standardized)
                sat_hist = self._extract_sat_hist(standardized)
                v_contour = self._extract_main_contour(standardized)
                v_edge = self._extract_edge_map(standardized)
                if hist is not None:
                    hist_list.append(hist)
                if sat_hist is not None:
                    sat_hist_list.append(sat_hist)
                variant_features.append((variant_code, hist, sat_hist, v_contour, v_edge))
            if hist_list:
                hist_avg = np.mean(np.stack(hist_list, axis=0), axis=0)
                cv2.normalize(hist_avg, hist_avg, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                self._base_template_hists[piece_type] = hist_avg
            if sat_hist_list:
                sat_hist_avg = np.mean(np.stack(sat_hist_list, axis=0), axis=0)
                cv2.normalize(sat_hist_avg, sat_hist_avg, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                self._base_template_sat_hists[piece_type] = sat_hist_avg
            if variant_features:
                self._type_variant_features[piece_type] = variant_features

    @staticmethod
    def _calc_feature_score(
            grid_hist, grid_sat_hist, contour, grid_edge,
            template_hist, template_sat_hist, template_contour, template_edge):
        hue_score = 1.0
        sat_score = 1.0
        shape_score = 1.0
        edge_score = 1.0

        if grid_hist is not None and template_hist is not None:
            hue_score = cv2.compareHist(grid_hist, template_hist, cv2.HISTCMP_BHATTACHARYYA)
        if grid_sat_hist is not None and template_sat_hist is not None:
            sat_score = cv2.compareHist(grid_sat_hist, template_sat_hist, cv2.HISTCMP_BHATTACHARYYA)

        if contour is not None and template_contour is not None:
            shape_score = cv2.matchShapes(contour, template_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            shape_score = min(1.0, float(shape_score) * 2.0)

        if grid_edge is not None and template_edge is not None:
            if template_edge.shape == grid_edge.shape:
                edge_score = cv2.norm(grid_edge, template_edge, cv2.NORM_L1) / (255.0 * grid_edge.size)
            else:
                resized_template_edge = cv2.resize(template_edge, (grid_edge.shape[1], grid_edge.shape[0]))
                edge_score = cv2.norm(grid_edge, resized_template_edge, cv2.NORM_L1) / (255.0 * grid_edge.size)

        return HUE_WEIGHT * hue_score + SAT_WEIGHT * sat_score + SHAPE_WEIGHT * shape_score + EDGE_WEIGHT * edge_score

    def classify_grids(self, grids):
        labels = np.zeros(len(grids), dtype=np.int32)
        subclass_labels = np.zeros(len(grids), dtype=np.int32)
        scores = np.ones(len(grids), dtype=np.float32)

        if not self._base_template_hists:
            return labels, subclass_labels, scores

        for i, grid in enumerate(grids):
            # Keep runtime grid in native appearance; only templates are standardized.
            grid_hist, grid_sat_hist, contour, grid_edge = self._extract_grid_features_once(grid)
            if grid_hist is None and grid_sat_hist is None and contour is None:
                labels[i] = 0
                subclass_labels[i] = 0
                scores[i] = 1.0
                continue

            best_type = 1
            best_score = float("inf")

            for piece_type in range(1, 7):
                score = self._calc_feature_score(
                    grid_hist, grid_sat_hist, contour, grid_edge,
                    self._base_template_hists.get(piece_type),
                    self._base_template_sat_hists.get(piece_type),
                    self._base_template_contours.get(piece_type),
                    self._base_template_edges.get(piece_type)
                )
                if score < best_score:
                    best_score = float(score)
                    best_type = piece_type

            best_variant_code = best_type * 10 + 1
            best_variant_score = float("inf")
            for variant_code, v_hist, v_sat_hist, v_contour, v_edge in self._type_variant_features.get(best_type, []):
                variant_score = self._calc_feature_score(
                    grid_hist, grid_sat_hist, contour, grid_edge,
                    v_hist, v_sat_hist, v_contour, v_edge
                )
                if variant_score < best_variant_score:
                    best_variant_score = float(variant_score)
                    best_variant_code = int(variant_code)

            labels[i] = best_type - 1
            subclass_labels[i] = best_variant_code
            scores[i] = float(min(1.0, best_score))

        return labels, subclass_labels, scores


class MatchThreeAgent:
    ROW_NUM = 8
    COL_NUM = 8

    def __init__(self, top_left, bottom_right, elem_images, action_delay=DEFAULT_ACTION_DELAY,
                 auto_recalibrate=DEFAULT_AUTO_RECALIBRATE,
                 recalibrate_interval=DEFAULT_RECALIBRATE_INTERVAL,
                 confidence_pause_threshold=DEFAULT_CONFIDENCE_PAUSE_THRESHOLD,
                 confidence_pause_streak=DEFAULT_CONFIDENCE_PAUSE_STREAK,
                 confidence_resume_threshold=DEFAULT_CONFIDENCE_RESUME_THRESHOLD,
                 confidence_resume_streak=DEFAULT_CONFIDENCE_RESUME_STREAK,
                 recalibrate_cooldown_cycles=DEFAULT_RECALIBRATE_COOLDOWN_CYCLES):
        self.prev_elem_array = None
        self.elem_array = np.zeros((8, 8), dtype=np.int64)
        self.elem_sub_array = np.zeros((8, 8), dtype=np.int64)
        self._grid_height = None
        self._grid_width = None
        self._x_edges = None
        self._y_edges = None
        self._board_shape = None
        self._grid_location = {}
        self._top_left = top_left
        self._bottom_right = bottom_right
        self._scores = [1] * self.ROW_NUM * self.COL_NUM
        self._action_delay = action_delay
        self._lookahead_discount = 0.72
        self._rollout_samples = 8
        self._rollout_topk = 4
        self._root_evaluation_cap = 6
        self._auto_recalibrate = bool(auto_recalibrate)
        self._recalibrate_interval = max(1, int(recalibrate_interval))
        self._cycle_index = 0
        self._ineffective_streak = 0
        self._perf_tracker = None
        self._sim_cache_max = 12000
        self._sim_cache = {}
        self._confidence_pause_threshold = float(confidence_pause_threshold)
        self._confidence_pause_streak = max(1, int(confidence_pause_streak))
        self._confidence_resume_threshold = min(
            self._confidence_pause_threshold,
            float(confidence_resume_threshold)
        )
        self._confidence_resume_streak = max(1, int(confidence_resume_streak))
        self._high_deviation_streak = 0
        self._low_deviation_streak = 0
        self._deviation_paused = False
        self._recalibrate_cooldown_cycles = max(1, int(recalibrate_cooldown_cycles))
        self._last_recalibrate_cycle = -10**9
        # self._elem_images = elem_images
        self._element_matcher = ElementMatcher(elem_images)

    def set_perf_tracker(self, perf_tracker):
        self._perf_tracker = perf_tracker

    def identify_game_board(self):
        x, y, w, h = self._top_left[0], self._top_left[1], self._bottom_right[0], self._bottom_right[1]
        return capture_screen_array(region=(x, y, w - x, h - y))

    def get_board_signature(self):
        sig = zlib.crc32(np.ascontiguousarray(self.elem_array).view(np.uint8))
        sig = zlib.crc32(np.ascontiguousarray(self.elem_sub_array).view(np.uint8), sig)
        return int(sig)

    @staticmethod
    def _line_has_triplet(line):
        run_len = 1
        prev = int(line[0])
        for value in line[1:]:
            cur = int(value)
            if cur != 0 and cur == prev:
                run_len += 1
                if run_len >= 3:
                    return True
            else:
                run_len = 1
                prev = cur
        return False

    def _has_match_after_swap_local(self, board, index1, index2):
        r1, c1 = index1
        r2, c2 = index2
        board[index1], board[index2] = board[index2], board[index1]

        has_match = False
        for row in (r1, r2):
            if self._line_has_triplet(board[row, :]):
                has_match = True
                break
        if not has_match:
            for col in (c1, c2):
                if self._line_has_triplet(board[:, col]):
                    has_match = True
                    break

        board[index1], board[index2] = board[index2], board[index1]
        return has_match

    def _update_grid_locations(self):
        if self._x_edges is None or self._y_edges is None:
            return
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                start_x, end_x = int(self._x_edges[j]), int(self._x_edges[j + 1])
                start_y, end_y = int(self._y_edges[i]), int(self._y_edges[i + 1])
                center_x = int(round(self._top_left[0] + (start_x + end_x) / 2.0))
                center_y = int(round(self._top_left[1] + (start_y + end_y) / 2.0))
                self._grid_location[(i, j)] = (center_x, center_y)

    def refresh_board_state(self):
        t_cap = time.perf_counter()
        board_array = self.identify_game_board()
        if self._perf_tracker is not None:
            self._perf_tracker.add("capture_board", time.perf_counter() - t_cap)

        t_split = time.perf_counter()
        grids = self.split_board_into_grids(board_array)
        if self._perf_tracker is not None:
            self._perf_tracker.add("split_grids", time.perf_counter() - t_split)

        t_update = time.perf_counter()
        self.update_elements(grids)
        if self._perf_tracker is not None:
            self._perf_tracker.add("update_elements", time.perf_counter() - t_update)
        return self.get_confidence_score()

    def should_take_action(self, wait_static):
        t_gate = time.perf_counter()
        board_static = (self.prev_elem_array is None or np.array_equal(self.prev_elem_array, self.elem_array))
        should_act = (not wait_static) or board_static
        if self._perf_tracker is not None:
            self._perf_tracker.add("wait_static_gate", time.perf_counter() - t_gate)
        return should_act

    def should_pause_for_high_deviation(self, avg_confidence_score):
        score = float(avg_confidence_score)

        if self._deviation_paused:
            if score <= self._confidence_resume_threshold:
                self._low_deviation_streak += 1
            else:
                self._low_deviation_streak = 0
            if self._low_deviation_streak >= self._confidence_resume_streak:
                self._deviation_paused = False
                self._high_deviation_streak = 0
                self._low_deviation_streak = 0
                return False
            return True

        if score >= self._confidence_pause_threshold:
            self._high_deviation_streak += 1
        else:
            self._high_deviation_streak = 0
        if self._high_deviation_streak >= self._confidence_pause_streak:
            self._deviation_paused = True
            self._low_deviation_streak = 0
            return True
        return False

    def should_attempt_recalibrate(self, avg_confidence_score):
        if not self._auto_recalibrate:
            return False
        if (self._cycle_index - self._last_recalibrate_cycle) < self._recalibrate_cooldown_cycles:
            return False
        score = float(avg_confidence_score)
        should_recalibrate = False
        if score >= DEFAULT_RECALIBRATE_BAD_SCORE:
            should_recalibrate = True
        elif self._ineffective_streak >= DEFAULT_RECALIBRATE_INEFFECTIVE_STREAK:
            should_recalibrate = True
        elif self._cycle_index % self._recalibrate_interval == 0 and score >= DEFAULT_RECALIBRATE_PERIODIC_SCORE:
            should_recalibrate = True

        if should_recalibrate:
            self._last_recalibrate_cycle = self._cycle_index
            return True
        return False

    def _split_board_into_grids_local(self, board_array):
        h, w, _ = board_array.shape
        x_edges = np.rint(np.linspace(0, w, self.COL_NUM + 1)).astype(np.int32)
        y_edges = np.rint(np.linspace(0, h, self.ROW_NUM + 1)).astype(np.int32)
        grids = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                sx, ex = int(x_edges[j]), int(x_edges[j + 1])
                sy, ey = int(y_edges[i]), int(y_edges[i + 1])
                grids.append(board_array[sy:ey, sx:ex])
        return grids

    def _count_action_candidates_on_board(self, board_array):
        self.elem_array = board_array
        return len(self.get_action())

    def try_recalibrate_region(self, current_score):
        x1, y1 = self._top_left
        x2, y2 = self._bottom_right
        board_w, board_h = x2 - x1, y2 - y1
        if board_w < 100 or board_h < 100:
            return None

        pad = DEFAULT_RECALIBRATE_PADDING
        origin_x = max(0, x1 - pad)
        origin_y = max(0, y1 - pad)
        cap_w = board_w + pad * 2
        cap_h = board_h + pad * 2

        t_cap = time.perf_counter()
        search_image = capture_screen_array(region=(origin_x, origin_y, cap_w, cap_h))
        if self._perf_tracker is not None:
            self._perf_tracker.add("recalibrate_capture", time.perf_counter() - t_cap)

        shifts = [
            (0, 0),
            (-DEFAULT_RECALIBRATE_STEP, 0), (DEFAULT_RECALIBRATE_STEP, 0),
            (0, -DEFAULT_RECALIBRATE_STEP), (0, DEFAULT_RECALIBRATE_STEP),
            (-DEFAULT_RECALIBRATE_STEP, -DEFAULT_RECALIBRATE_STEP),
            (-DEFAULT_RECALIBRATE_STEP, DEFAULT_RECALIBRATE_STEP),
            (DEFAULT_RECALIBRATE_STEP, -DEFAULT_RECALIBRATE_STEP),
            (DEFAULT_RECALIBRATE_STEP, DEFAULT_RECALIBRATE_STEP),
        ]

        current_board = self.elem_array.copy()
        best_quality = -float(current_score)
        best_shift = (0, 0)
        best_score = float(current_score)

        t_eval = time.perf_counter()
        for dx, dy in shifts:
            sx = (x1 + dx) - origin_x
            sy = (y1 + dy) - origin_y
            ex = sx + board_w
            ey = sy + board_h
            if sx < 0 or sy < 0 or ex > search_image.shape[1] or ey > search_image.shape[0]:
                continue

            board = search_image[sy:ey, sx:ex]
            grids = self._split_board_into_grids_local(board)
            labels, _, scores = self._element_matcher.classify_grids(grids)
            candidate_elem = labels.reshape((self.ROW_NUM, self.COL_NUM)) + 1
            candidate_score = float(np.mean(scores))
            actions_count = self._count_action_candidates_on_board(candidate_elem.copy())
            candidate_quality = -candidate_score + min(40, actions_count) * 0.0015

            if candidate_quality > best_quality:
                best_quality = candidate_quality
                best_shift = (dx, dy)
                best_score = candidate_score

        self.elem_array = current_board
        if self._perf_tracker is not None:
            self._perf_tracker.add("recalibrate_evaluate", time.perf_counter() - t_eval)

        if best_shift == (0, 0):
            return None
        if (float(current_score) - best_score) < 0.006:
            return None

        self._top_left = (x1 + best_shift[0], y1 + best_shift[1])
        self._bottom_right = (x2 + best_shift[0], y2 + best_shift[1])
        self._update_grid_locations()
        print(f"Auto recalibrated board shift dx={best_shift[0]}, dy={best_shift[1]}, score {current_score:.4f}->{best_score:.4f}")
        return best_score

    @staticmethod
    def normalize_action(action):
        if not action:
            return None
        return (
            (int(action[0][0]), int(action[0][1])),
            (int(action[1][0]), int(action[1][1]))
        )

    def wait_until_board_settled(
            self,
            timeout=DEFAULT_SETTLE_TIMEOUT,
            poll_interval=DEFAULT_SETTLE_POLL_INTERVAL,
            stable_frames=DEFAULT_SETTLE_STABLE_FRAMES,
            diff_threshold=DEFAULT_SETTLE_DIFF_THRESHOLD):
        start_ts = time.time()
        prev_gray = None
        stable_count = 0

        while time.time() - start_ts < timeout:
            board = self.identify_game_board()
            # Downscale + center crop + blur for robust settle check against UI particles.
            small_board = cv2.resize(
                board,
                (DEFAULT_SETTLE_DOWNSCALE, DEFAULT_SETTLE_DOWNSCALE),
                interpolation=cv2.INTER_AREA
            )
            h, w = small_board.shape[:2]
            margin_h = max(1, int(h * 0.1))
            margin_w = max(1, int(w * 0.1))
            center_roi = small_board[margin_h:h - margin_h, margin_w:w - margin_w]
            cur_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.GaussianBlur(cur_gray, (3, 3), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(cur_gray, prev_gray)
                mean_diff = float(np.mean(diff))
                if mean_diff <= diff_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= stable_frames:
                    return True

            prev_gray = cur_gray
            time.sleep(poll_interval)

        return False

    def split_board_into_grids(self, grid_array):
        height, width, _ = grid_array.shape

        # Initialize grid location on the screen
        if self._board_shape != (height, width):
            self._board_shape = (height, width)
            self._x_edges = np.rint(np.linspace(0, width, self.COL_NUM + 1)).astype(np.int32)
            self._y_edges = np.rint(np.linspace(0, height, self.ROW_NUM + 1)).astype(np.int32)
            self._grid_width = int(round(width / self.COL_NUM))
            self._grid_height = int(round(height / self.ROW_NUM))
            self._element_matcher.set_grid_size(self._grid_width, self._grid_height)
            self._update_grid_locations()
            # for elem_name, image in self._elem_images.items():
            #     self._elem_images[elem_name] = cv2.resize(image, (self._grid_width, self._grid_height))  # unused

        grids = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                x_margin, y_margin = 0, 0
                start_x = int(self._x_edges[j] + x_margin)
                end_x = int(self._x_edges[j + 1] - x_margin)
                start_y = int(self._y_edges[i] + y_margin)
                end_y = int(self._y_edges[i + 1] - y_margin)
                grid = grid_array[start_y:end_y, start_x:end_x]
                grids.append(grid)

        return grids

    def update_elements(self, grids):
        labels, subclass_labels, scores = self._element_matcher.classify_grids(grids)

        elem_array = np.zeros((8, 8), dtype=np.int64)
        elem_sub_array = np.zeros((8, 8), dtype=np.int64)

        self._scores = []
        for i, (label, subclass_label, score) in enumerate(zip(labels, subclass_labels, scores)):
            row, col = i // 8, i % 8
            elem_array[row, col] = int(label) + 1
            elem_sub_array[row, col] = int(subclass_label)
            self._scores.append(float(score))

        self.prev_elem_array = self.elem_array
        self.elem_array = elem_array
        self.elem_sub_array = elem_sub_array

    def force_resync_board_state(self, samples=DEFAULT_RECOGNITION_RETRY_SAMPLES, interval=DEFAULT_RECOGNITION_RETRY_INTERVAL):
        best = None
        for i in range(max(1, int(samples))):
            board_array = self.identify_game_board()
            grids = self.split_board_into_grids(board_array)
            labels, subclass_labels, scores = self._element_matcher.classify_grids(grids)

            elem_array = np.zeros((8, 8), dtype=np.int64)
            elem_sub_array = np.zeros((8, 8), dtype=np.int64)
            score_list = []
            for k, (label, subclass_label, score) in enumerate(zip(labels, subclass_labels, scores)):
                row, col = k // 8, k % 8
                elem_array[row, col] = int(label) + 1
                elem_sub_array[row, col] = int(subclass_label)
                score_list.append(float(score))

            avg_score = float(np.mean(score_list)) if score_list else 1.0
            if best is None or avg_score < best[0]:
                best = (avg_score, elem_array, elem_sub_array, score_list)

            if i + 1 < samples:
                time.sleep(max(0.0, float(interval)))

        if best is not None:
            _, elem_array, elem_sub_array, score_list = best
            self.prev_elem_array = None
            self.elem_array = elem_array
            self.elem_sub_array = elem_sub_array
            self._scores = score_list
        return self.get_confidence_score()

    def get_grid_element(self, index):
        if 0 <= index[0] < self.ROW_NUM and 0 <= index[1] < self.COL_NUM:
            return self.elem_array[index]
        return None

    def grid_index_add(self, index1, index2):
        return index1[0] + index2[0], index1[1] + index2[1]

    def grid_index_subtract(self, index1, index2):
        return index1[0] - index2[0], index1[1] - index2[1]

    def get_action(self):
        def equal_match_value(index1, index2):
            return index1 in cur_grid_lst and index2 in cur_grid_lst and ar_match[index1] == ar_match[index2]

        def index_could_swap(index):
            return index in cur_grid_lst and ar_swap[index]

        def normalized_pair(index1, index2):
            a = (int(index1[0]), int(index1[1]))
            b = (int(index2[0]), int(index2[1]))
            return (a, b) if a <= b else (b, a)

        def add_action(index1, index2):
            action = ((int(index1[0]), int(index1[1])), (int(index2[0]), int(index2[1])))
            key = normalized_pair(action[0], action[1])
            if key not in action_map:
                # Keep first-discovered direction to avoid random reversal from set ordering.
                action_map[key] = action

        cur_grid_lst = [(i, j) for i in range(0, self.ROW_NUM) for j in range(0, self.COL_NUM)]
        ar_swap = np.zeros((self.ROW_NUM, self.COL_NUM))
        ar_match = np.zeros((self.ROW_NUM, self.COL_NUM))
        for index in cur_grid_lst:
            ar_swap[index] = self.elem_array[index]
            ar_match[index] = self.elem_array[index]

        action_map = {}
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-1] == ar_match[:, 1:], ar_match[:, :-1] != 0)):
            if index_could_swap((i, j - 1)):
                for index in [(i - 1, j - 1), (i + 1, j - 1), (i, j - 2)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        add_action((i, j - 1), index)
            if index_could_swap((i, j + 2)):
                for index in [(i - 1, j + 2), (i + 1, j + 2), (i, j + 3)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        add_action((i, j + 2), index)
        for i, j in np.argwhere(np.logical_and(ar_match[:-1, :] == ar_match[1:, :], ar_match[:-1, :] != 0)):
            if index_could_swap((i - 1, j)):
                for index in [(i - 1, j - 1), (i - 1, j + 1), (i - 2, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        add_action((i - 1, j), index)
            if index_could_swap((i + 2, j)):
                for index in [(i + 2, j - 1), (i + 2, j + 1), (i + 3, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        add_action((i + 2, j), index)
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-2] == ar_match[:, 2:], ar_match[:, :-2] != 0)):
            if index_could_swap((i, j + 1)):
                for index in [(i - 1, j + 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            add_action((i, index[1]), index)
        for i, j in np.argwhere(np.logical_and(ar_match[:-2, :] == ar_match[2:, :], ar_match[:-2, :] != 0)):
            if index_could_swap((i + 1, j)):
                for index in [(i + 1, j - 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            add_action((index[0], j), index)

        return list(action_map.values())

    def _find_matches(self, board):
        horizontal_groups = []
        vertical_groups = []
        matched_cells = set()

        # Horizontal runs
        for i in range(self.ROW_NUM):
            j = 0
            while j < self.COL_NUM:
                elem = board[i, j]
                if elem == 0:
                    j += 1
                    continue
                k = j + 1
                while k < self.COL_NUM and board[i, k] == elem:
                    k += 1
                run_len = k - j
                if run_len >= 3:
                    group = [(i, c) for c in range(j, k)]
                    horizontal_groups.append(group)
                    matched_cells.update(group)
                j = k

        # Vertical runs
        for j in range(self.COL_NUM):
            i = 0
            while i < self.ROW_NUM:
                elem = board[i, j]
                if elem == 0:
                    i += 1
                    continue
                k = i + 1
                while k < self.ROW_NUM and board[k, j] == elem:
                    k += 1
                run_len = k - i
                if run_len >= 3:
                    group = [(r, j) for r in range(i, k)]
                    vertical_groups.append(group)
                    matched_cells.update(group)
                i = k

        return {
            "matched_cells": matched_cells,
            "horizontal_groups": horizontal_groups,
            "vertical_groups": vertical_groups,
        }

    def _classify_match_priority(self, match_info):
        horizontal_groups = match_info["horizontal_groups"]
        vertical_groups = match_info["vertical_groups"]
        matched_cells = match_info["matched_cells"]

        if not matched_cells:
            return "invalid", None

        horizontal_cells = set()
        vertical_cells = set()
        for group in horizontal_groups:
            horizontal_cells.update(group)
        for group in vertical_groups:
            vertical_cells.update(group)

        # 3+3 cross creates at the intersection.
        cross_points = horizontal_cells.intersection(vertical_cells)
        if cross_points:
            cross = sorted(cross_points)[0]
            return "cross_3_3", cross

        # 5-line creates at center.
        for group in horizontal_groups + vertical_groups:
            if len(group) >= 5:
                center = group[len(group) // 2]
                return "line_5", center

        # 4-line horizontal creates at the 2nd cell from left to right.
        for group in horizontal_groups:
            if len(group) >= 4:
                return "line_4_h", group[1]

        # Vertical 4 has no meaningful retained special by requested rule.
        return "line_3", None

    def _priority_base_score(self, priority_type):
        if priority_type == "cross_3_3":
            return 20000
        if priority_type == "line_5":
            return 16000
        if priority_type == "line_4_h":
            return 9000
        if priority_type == "line_3":
            return 3000
        return -10**9

    def _priority_rank(self, priority_type):
        if priority_type in ("cross_3_3", "line_5"):
            return 3
        if priority_type == "line_4_h":
            return 2
        if priority_type == "line_3":
            return 1
        return 0

    @staticmethod
    def _advanced_piece_tier(sub_code):
        sub_code = int(sub_code)
        variant = sub_code % 10
        if sub_code == 43:
            return 5
        if sub_code == 63:
            return 4
        if sub_code == 22:
            return 3
        if variant == 3:
            return 2
        if variant == 2:
            return 1
        return 0

    @staticmethod
    def _advanced_piece_score(sub_code):
        tier = MatchThreeAgent._advanced_piece_tier(sub_code)
        if tier == 5:
            return 1200
        if tier == 4:
            return 950
        if tier == 3:
            return 780
        if tier == 2:
            return 460
        if tier == 1:
            return 260
        return 0

    @staticmethod
    def _advanced_competitive_rank(tier):
        # Only top-2 advanced tiers keep strict global priority.
        # Tier-3+ joins normal match-pattern competition.
        if tier >= 4:
            return 0
        if tier == 3:
            # 2_2 competes around 4-line level.
            return 2
        if tier == 2:
            # other x_3 competes around 3-line/4-line boundary (weaker than tier-3).
            return 1
        if tier == 1:
            # other x_2 has mild tactical preference.
            return 1
        return 0

    @staticmethod
    def _predict_generated_sub_code(simulated_board, match_type, create_pos):
        if create_pos is None:
            return 0
        if match_type in ("cross_3_3", "line_5"):
            variant = 3
        elif match_type == "line_4_h":
            variant = 2
        else:
            return 0

        r, c = int(create_pos[0]), int(create_pos[1])
        if r < 0 or c < 0 or r >= simulated_board.shape[0] or c >= simulated_board.shape[1]:
            return 0
        piece_type = int(simulated_board[r, c])
        if piece_type <= 0:
            return 0
        return piece_type * 10 + variant

    def _action_priority_key(self, board, sub_board, action):
        simulated = board.copy()
        index1, index2 = action
        simulated[index1], simulated[index2] = simulated[index2], simulated[index1]
        match_info = self._find_matches(simulated)
        matched_cells = match_info["matched_cells"]
        if not matched_cells:
            return (-1, -1, -1, -1, -1.0, -1)

        # 1) Advanced-piece elimination priority.
        best_advanced_tier = 0
        advanced_score = 0
        for row, col in matched_cells:
            sub_code = int(sub_board[row, col]) if sub_board is not None else 0
            best_advanced_tier = max(best_advanced_tier, self._advanced_piece_tier(sub_code))
            advanced_score += self._advanced_piece_score(sub_code)

        # 2) Match pattern priority (5/cross > 4_h > 3).
        match_type, create_pos = self._classify_match_priority(match_info)
        match_rank = self._priority_rank(match_type)

        # Only top-2 advanced tiers (4_3 / 6_3) are strict global priority.
        elite_advanced_rank = best_advanced_tier if best_advanced_tier >= 4 else 0
        competitive_advanced_rank = self._advanced_competitive_rank(best_advanced_tier)
        competitive_rank = max(match_rank, competitive_advanced_rank)

        # Special-piece generation preference under same 5/4 level.
        generated_sub_code = self._predict_generated_sub_code(simulated, match_type, create_pos)
        generated_tier = self._advanced_piece_tier(generated_sub_code)
        generated_score = self._advanced_piece_score(generated_sub_code)

        # 3) Direction preference: vertical > horizontal (for same higher-level conditions).
        has_vertical = any(len(g) >= 3 for g in match_info["vertical_groups"])
        has_horizontal = any(len(g) >= 3 for g in match_info["horizontal_groups"])
        if has_vertical and not has_horizontal:
            direction_rank = 2
        elif has_vertical:
            direction_rank = 1
        else:
            direction_rank = 0

        # 4) Lower-board preference.
        bottom_rank = float(sum(r for r, _ in matched_cells)) / max(1, len(matched_cells))
        largest_row = max(r for r, _ in matched_cells)

        return (
            elite_advanced_rank,
            competitive_rank,
            match_rank,
            generated_tier,
            generated_score,
            advanced_score,
            competitive_advanced_rank,
            direction_rank,
            bottom_rank,
            largest_row
        )

    def _apply_gravity_and_fill(self, board, rng, fill_values):
        new_board = board.copy()
        for col in range(self.COL_NUM):
            non_zero = [new_board[row, col] for row in range(self.ROW_NUM) if new_board[row, col] != 0]
            for row in range(self.ROW_NUM - 1, -1, -1):
                if non_zero:
                    new_board[row, col] = non_zero.pop()
                else:
                    new_board[row, col] = rng.choice(fill_values)
        return new_board

    def _simulate_action(self, board, action, depth):
        cache_key = None
        if self._sim_cache is not None:
            cache_key = (board.tobytes(), action, int(depth))
            cached = self._sim_cache.get(cache_key)
            if cached is not None:
                cached_score, cached_board = cached
                return cached_score, (None if cached_board is None else cached_board.copy())

        index1, index2 = action
        if index1 == index2:
            return -10**9, None
        if abs(index1[0] - index2[0]) + abs(index1[1] - index2[1]) != 1:
            return -10**9, None

        simulated = board.copy()
        simulated[index1], simulated[index2] = simulated[index2], simulated[index1]

        initial_match = self._find_matches(simulated)
        if not initial_match["matched_cells"]:
            return -10**9, None

        priority_type, _ = self._classify_match_priority(initial_match)
        score = self._priority_base_score(priority_type)

        non_zero_values = [int(v) for v in board.flatten() if v != 0]
        fill_values = non_zero_values if non_zero_values else list(range(1, 7))
        action_seed = int(
            index1[0] * 10007 + index1[1] * 10009 + index2[0] * 10037 + index2[1] * 10039 + depth * 10061
        ) & 0xffffffff
        rng = random.Random(action_seed)

        chain = 0
        while True:
            match_info = self._find_matches(simulated)
            matched_cells = match_info["matched_cells"]
            if not matched_cells:
                break

            chain += 1
            cleared = len(matched_cells)
            score += cleared * 65
            if chain > 1:
                score += chain * 180

            for row, col in matched_cells:
                simulated[row, col] = 0
            simulated = self._apply_gravity_and_fill(simulated, rng, fill_values)

            if chain >= 8:
                break

        if cache_key is not None:
            if len(self._sim_cache) >= self._sim_cache_max:
                self._sim_cache.clear()
            self._sim_cache[cache_key] = (score, simulated.copy())
        return score, simulated

    def _generate_valid_swap_actions(self, board):
        actions = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                for ni, nj in ((i + 1, j), (i, j + 1)):
                    if ni >= self.ROW_NUM or nj >= self.COL_NUM:
                        continue
                    if self._has_match_after_swap_local(board, (i, j), (ni, nj)):
                        actions.append(((i, j), (ni, nj)))
        return actions

    def _evaluate_action_expectimax_lite(self, board, action, depth):
        immediate_score, next_board = self._simulate_action(board, action, depth)
        if next_board is None or depth <= 1:
            return immediate_score

        # Deterministic seed for stable behavior under the same board/action.
        idx1, idx2 = action
        seed_base = int(
            idx1[0] * 10007 + idx1[1] * 10009 + idx2[0] * 10037 + idx2[1] * 10039 + depth * 10061
        ) & 0xffffffff

        rollout_sum = 0.0
        valid_action_cache = {}
        for sample_idx in range(self._rollout_samples):
            rng = random.Random(seed_base + sample_idx * 7919)
            cur_board = next_board.copy()
            sample_score = 0.0

            for d in range(depth - 1):
                board_key = cur_board.tobytes()
                actions = valid_action_cache.get(board_key)
                if actions is None:
                    actions = tuple(self._generate_valid_swap_actions(cur_board))
                    valid_action_cache[board_key] = actions
                if not actions:
                    break

                scored = []
                remain_depth = max(1, depth - 1 - d)
                for cand in actions:
                    s, nb = self._simulate_action(cur_board, cand, remain_depth)
                    if nb is not None:
                        scored.append((s, cand, nb))
                if not scored:
                    break

                scored.sort(key=lambda x: x[0], reverse=True)
                topk = min(self._rollout_topk, len(scored))
                chosen_score, _, chosen_board = scored[rng.randrange(topk)]
                sample_score += (self._lookahead_discount ** d) * chosen_score
                cur_board = chosen_board

            rollout_sum += sample_score

        expected_future = rollout_sum / max(1, self._rollout_samples)
        return immediate_score + self._lookahead_discount * expected_future

    def choose_best_action(self, actions, lookahead_depth=3, banned_actions=None):
        self._sim_cache.clear()
        banned_actions = banned_actions or set()
        self_actions = []

        candidate_actions = []
        for action in actions:
            if not action:
                continue
            index1, index2 = action
            index1 = (int(index1[0]), int(index1[1]))
            index2 = (int(index2[0]), int(index2[1]))
            if index1 == index2:
                self_actions.append((tuple(index1), tuple(index2)))
                continue
            if abs(index1[0] - index2[0]) + abs(index1[1] - index2[1]) != 1:
                continue
            normalized_action = (tuple(index1), tuple(index2))
            if normalized_action in banned_actions:
                continue
            candidate_actions.append(normalized_action)

        if not candidate_actions:
            if self_actions:
                return random.choice(self_actions)
            return []

        board = self.elem_array.copy()
        sub_board = self.elem_sub_array.copy() if self.elem_sub_array is not None else None

        # Enforce strict priority order:
        # advanced-piece elimination > (5/cross > 4_h > 3) > vertical > lower-board.
        ranked_candidates = []
        for action in candidate_actions:
            key = self._action_priority_key(board, sub_board, action)
            ranked_candidates.append((key, action))
        ranked_candidates.sort(key=lambda x: x[0], reverse=True)

        # Keep a small frontier of best rule-ranked actions for simulation.
        candidate_actions = [action for _, action in ranked_candidates[:max(self._root_evaluation_cap * 2, 8)]]

        # Root pruning by immediate score to keep runtime bounded.
        immediate_ranked = []
        for action in candidate_actions:
            s, _ = self._simulate_action(board, action, max(1, lookahead_depth))
            immediate_ranked.append((s, action))
        immediate_ranked.sort(key=lambda x: x[0], reverse=True)
        candidate_actions = [a for _, a in immediate_ranked[:self._root_evaluation_cap]]

        best_action = None
        best_score = -10**9
        for action in candidate_actions:
            score = self._evaluate_action_expectimax_lite(board, action, max(1, lookahead_depth))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else []

    def take_action(self, action):
        def swap_element(index1, index2):
            if index1 == index2:
                return

            # Conservative drag: press firmly first, then move, then release.
            pyautogui.moveTo(index1[0], index1[1], duration=0.05)
            time.sleep(0.012)
            pyautogui.mouseDown()
            time.sleep(max(DEFAULT_DRAG_HOLD_BEFORE_MOVE, 0.035))
            pyautogui.moveTo(index2[0], index2[1], duration=0.18)
            time.sleep(max(DEFAULT_DRAG_HOLD_AFTER_MOVE, 0.060))
            pyautogui.mouseUp()

        if not action:
            return []

        screen_index1, screen_index2 = self._grid_location[action[0]], self._grid_location[action[1]]
        swap_element(screen_index1, screen_index2)
        # Give the board a short settle time to avoid issuing the next move too early.
        time.sleep(self._action_delay)
        return action

    def retry_action(self, action):
        if not action:
            return False
        screen_index1, screen_index2 = self._grid_location[action[0]], self._grid_location[action[1]]

        # Fallback: same conservative drag with longer hold.
        pyautogui.moveTo(screen_index1[0], screen_index1[1], duration=0.06)
        time.sleep(0.015)
        pyautogui.mouseDown()
        time.sleep(max(DEFAULT_DRAG_HOLD_BEFORE_MOVE, 0.045))
        pyautogui.moveTo(screen_index2[0], screen_index2[1], duration=0.24)
        time.sleep(max(DEFAULT_DRAG_HOLD_AFTER_MOVE_RETRY, 0.075))
        pyautogui.mouseUp()
        time.sleep(min(0.35, max(0.16, self._action_delay * 0.8)))
        return True

    def get_confidence_score(self):
        return np.mean(self._scores)


def run(wait_static, show, action_delay, lookahead_depth, disable_wait_settle, settle_timeout,
        perf=False, perf_every=DEFAULT_PERF_SUMMARY_EVERY,
        auto_recalibrate=DEFAULT_AUTO_RECALIBRATE,
        recalibrate_interval=DEFAULT_RECALIBRATE_INTERVAL,
        idle_poll_interval=DEFAULT_IDLE_POLL_INTERVAL,
        confidence_pause_threshold=DEFAULT_CONFIDENCE_PAUSE_THRESHOLD,
        confidence_pause_streak=DEFAULT_CONFIDENCE_PAUSE_STREAK,
        confidence_resume_threshold=DEFAULT_CONFIDENCE_RESUME_THRESHOLD,
        confidence_resume_streak=DEFAULT_CONFIDENCE_RESUME_STREAK,
        recalibrate_cooldown_cycles=DEFAULT_RECALIBRATE_COOLDOWN_CYCLES):
    wait_for_start_signal()

    cur_path = sys.argv[0]
    cur_dir = os.path.dirname(cur_path)
    resource_dir = os.path.join(cur_dir, 'resource')

    top_left, bottom_right = resolve_board_coordinates(resource_dir=resource_dir)
    show_board_if_needed(show, top_left, bottom_right)

    elem_images = load_elem_images(resource_dir)
    agent = MatchThreeAgent(
        top_left,
        bottom_right,
        elem_images,
        action_delay=action_delay,
        auto_recalibrate=auto_recalibrate,
        recalibrate_interval=recalibrate_interval,
        confidence_pause_threshold=confidence_pause_threshold,
        confidence_pause_streak=confidence_pause_streak,
        confidence_resume_threshold=confidence_resume_threshold,
        confidence_resume_streak=confidence_resume_streak,
        recalibrate_cooldown_cycles=recalibrate_cooldown_cycles
    )
    perf_tracker = PerfTracker(enabled=perf, summary_every=perf_every)
    agent.set_perf_tracker(perf_tracker)
    last_board_signature = None
    last_action = None

    try:
        while True:
            should_stop, last_board_signature, last_action = run_main_cycle(
                agent=agent,
                wait_static=wait_static,
                lookahead_depth=lookahead_depth,
                disable_wait_settle=disable_wait_settle,
                settle_timeout=settle_timeout,
                last_board_signature=last_board_signature,
                last_action=last_action,
                perf_tracker=perf_tracker,
                idle_poll_interval=idle_poll_interval
            )
            if should_stop:
                break
    finally:
        perf_tracker.print_summary()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Auto-play Dota2 Dragon Chess using visual board recognition."
    )

    common = parser.add_argument_group("Common")
    common.add_argument('-w', '--wait_static', help='wait for a static board before acting', action="store_true")
    common.add_argument('-s', '--show', '--show_board', dest='show_board',
                        help='show the identified board image for debugging', action="store_true")
    common.add_argument('--action_delay', type=float, default=DEFAULT_ACTION_DELAY,
                        help=f'delay after each swap action in seconds (default: {DEFAULT_ACTION_DELAY})')
    common.add_argument('--idle_poll_interval', type=float, default=DEFAULT_IDLE_POLL_INTERVAL,
                        help=f'poll interval in seconds for paused/idle cycles (default: {DEFAULT_IDLE_POLL_INTERVAL})')

    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument('--lookahead_depth', type=int, default=DEFAULT_LOOKAHEAD_DEPTH,
                          help=f'search depth for move planning, recommended 3~4 (default: {DEFAULT_LOOKAHEAD_DEPTH})')
    advanced.add_argument('--disable_wait_settle', action="store_true",
                          help='disable waiting for board to settle after each swap')
    advanced.add_argument('--settle_timeout', type=float, default=DEFAULT_SETTLE_TIMEOUT,
                          help=f'max seconds to wait for board settling after each swap (default: {DEFAULT_SETTLE_TIMEOUT})')
    advanced.add_argument('--disable_auto_recalibrate', action="store_true",
                          help='disable auto recalibration based on board recognition feedback')
    advanced.add_argument('--recalibrate_interval', type=int, default=DEFAULT_RECALIBRATE_INTERVAL,
                          help=f'run auto recalibration every N cycles (default: {DEFAULT_RECALIBRATE_INTERVAL})')
    advanced.add_argument('--recalibrate_cooldown_cycles', type=int, default=DEFAULT_RECALIBRATE_COOLDOWN_CYCLES,
                          help='minimum cycle gap between two auto recalibration attempts')
    advanced.add_argument('--perf', action="store_true",
                          help='enable per-step runtime profiling and periodic summary logs')
    advanced.add_argument('--perf_every', type=int, default=DEFAULT_PERF_SUMMARY_EVERY,
                          help=f'print perf summary every N cycles (default: {DEFAULT_PERF_SUMMARY_EVERY})')
    advanced.add_argument('--confidence_pause_threshold', type=float, default=DEFAULT_CONFIDENCE_PAUSE_THRESHOLD,
                          help='treat recognition as paused when confidence deviation is above this threshold')
    advanced.add_argument('--confidence_pause_streak', type=int, default=DEFAULT_CONFIDENCE_PAUSE_STREAK,
                          help='consecutive high-deviation cycles required before pausing actions')
    advanced.add_argument('--confidence_resume_threshold', type=float, default=DEFAULT_CONFIDENCE_RESUME_THRESHOLD,
                          help='deviation threshold to exit auto-pause mode (hysteresis lower bound)')
    advanced.add_argument('--confidence_resume_streak', type=int, default=DEFAULT_CONFIDENCE_RESUME_STREAK,
                          help='consecutive low-deviation cycles required to exit auto-pause mode')
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    print("Move the mouse over 'play' button and press 'b' to start.")

    listener = Listener(on_press=on_press)
    listener.start()

    try:
        run(
            wait_static=args.wait_static,
            show=args.show_board,
            action_delay=max(0.0, args.action_delay),
            lookahead_depth=min(DEFAULT_LOOKAHEAD_DEPTH, max(1, args.lookahead_depth)),
            disable_wait_settle=args.disable_wait_settle,
            settle_timeout=max(0.3, args.settle_timeout),
            perf=args.perf,
            perf_every=max(1, args.perf_every),
            auto_recalibrate=(not args.disable_auto_recalibrate),
            recalibrate_interval=max(1, args.recalibrate_interval),
            recalibrate_cooldown_cycles=max(1, int(args.recalibrate_cooldown_cycles)),
            idle_poll_interval=max(0.05, args.idle_poll_interval),
            confidence_pause_threshold=max(0.0, float(args.confidence_pause_threshold)),
            confidence_pause_streak=max(1, int(args.confidence_pause_streak)),
            confidence_resume_threshold=max(0.0, float(args.confidence_resume_threshold)),
            confidence_resume_streak=max(1, int(args.confidence_resume_streak))
        )
    except KeyboardInterrupt:
        stop_agent = True
        print("KeyboardInterrupt received, exiting...")
    except Exception as e:
        print('An error occurred: {}'.format(e))
        time.sleep(5)
        raise
    finally:
        try:
            listener.stop()
        except Exception:
            pass
