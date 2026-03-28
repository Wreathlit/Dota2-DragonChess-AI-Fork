import os
import sys
import random
import time
import ctypes
from ctypes import wintypes
from collections import defaultdict
import argparse
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Listener, Key


start_agent = False
stop_agent = False
pause_agent = False
DEFAULT_WINDOW_TITLES = ["Dota 2"]
WINDOW_TITLE_EXCLUDES = ["visual studio code", "cursor", "pycharm", "notepad++", "sublime text"]
DEFAULT_MATCH_CONFIDENCE = 0.72
DEFAULT_ACTION_DELAY = 0.5
DEFAULT_LOOKAHEAD_DEPTH = 3
DEFAULT_SETTLE_TIMEOUT = 2.0
HSV_SAT_THRESHOLD = 40
HSV_VAL_THRESHOLD = 35
MORPH_KERNEL_SIZE = 3
MIN_CONTOUR_AREA = 8.0
HUE_HIST_BINS = 30
HUE_WEIGHT = 0.75
SHAPE_WEIGHT = 0.25


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


def capture_screen_array(region=None):
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


def _match_template_multiscale(search_image, target_image, confidence=0.72, scale_min=0.55, scale_max=1.45, scale_steps=19):
    search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    search_edge = cv2.Canny(search_gray, 40, 120)
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

    top_left_template = cv2.imread(os.path.join(resource_dir, 'topLeft.jpg'), cv2.IMREAD_COLOR)
    bottom_right_template = cv2.imread(os.path.join(resource_dir, 'botRight.jpg'), cv2.IMREAD_COLOR)
    if top_left_template is None or bottom_right_template is None:
        return None, None

    top_left_loc = _match_template_multiscale(image_array, top_left_template, confidence=confidence)
    bottom_right_loc = _match_template_multiscale(image_array, bottom_right_template, confidence=confidence)

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


def execute_action_cycle(agent, action, disable_wait_settle, settle_timeout):
    action = agent.take_action(action)
    if action and not disable_wait_settle:
        settled = agent.wait_until_board_settled(timeout=settle_timeout)
        if not settled:
            print("Board settle wait timeout; continue with next cycle.")
    return action


def print_cycle_log(elem_array, action, confidence_score, elapsed_sec):
    print(elem_array)
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
                   last_board_signature, last_action):
    if stop_agent:
        print("Program stopped by user.")
        return True, last_board_signature, last_action

    if pause_agent:
        print("Program paused by user. Press 'p' or 'q' again to unpause.")
        time.sleep(1)
        return False, last_board_signature, last_action

    t1 = time.time()
    avg_confidence_score = agent.refresh_board_state()
    should_act = agent.should_take_action(wait_static)
    if not should_act:
        return False, last_board_signature, last_action

    actions = agent.get_action()
    board_signature = agent.elem_array.tobytes()
    action, should_skip = choose_action_with_repeat_guard(
        agent=agent,
        actions=actions,
        board_signature=board_signature,
        last_board_signature=last_board_signature,
        last_action=last_action,
        lookahead_depth=lookahead_depth
    )
    if should_skip:
        time.sleep(0.08)
        return False, last_board_signature, last_action

    action = execute_action_cycle(
        agent=agent,
        action=action,
        disable_wait_settle=disable_wait_settle,
        settle_timeout=settle_timeout
    )
    print_cycle_log(
        elem_array=agent.elem_array,
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

    def _extract_main_contour(self, image):
        crop, mask = self._extract_piece_mask(image)
        if crop is None or mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            return None
        return contour

    def _extract_hue_hist(self, image):
        crop, mask = self._extract_piece_mask(image)
        if crop is None or mask is None:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if cv2.countNonZero(mask) < 10:
            return None

        hist = cv2.calcHist([hsv], [0], mask, [HUE_HIST_BINS], [0, 180])
        if hist is None:
            return None
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist

    def set_grid_size(self, width, height):
        self._grid_size = (width, height)
        self._base_template_contours = {}
        self._base_template_hists = {}

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

            # Hue template uses all variants for robustness.
            hist_list = []
            for _, img in variants:
                resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                hist = self._extract_hue_hist(resized)
                if hist is not None:
                    hist_list.append(hist)
            if hist_list:
                hist_avg = np.mean(np.stack(hist_list, axis=0), axis=0)
                cv2.normalize(hist_avg, hist_avg, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                self._base_template_hists[piece_type] = hist_avg

    def classify_grids(self, grids):
        labels = np.zeros(len(grids), dtype=np.int32)
        scores = np.ones(len(grids), dtype=np.float32)

        if not self._base_template_hists:
            return labels, scores

        for i, grid in enumerate(grids):
            grid_hist = self._extract_hue_hist(grid)
            contour = self._extract_main_contour(grid)
            if grid_hist is None and contour is None:
                labels[i] = 0
                scores[i] = 1.0
                continue

            best_type = 1
            best_score = float("inf")

            for piece_type in range(1, 7):
                hue_score = 1.0
                shape_score = 1.0

                if grid_hist is not None and piece_type in self._base_template_hists:
                    hue_score = cv2.compareHist(grid_hist, self._base_template_hists[piece_type], cv2.HISTCMP_BHATTACHARYYA)

                if contour is not None and piece_type in self._base_template_contours:
                    shape_score = cv2.matchShapes(contour, self._base_template_contours[piece_type], cv2.CONTOURS_MATCH_I1, 0.0)

                # Color first, shape as tie-break/fallback.
                score = HUE_WEIGHT * hue_score + SHAPE_WEIGHT * shape_score
                if score < best_score:
                    best_score = float(score)
                    best_type = piece_type

            labels[i] = best_type - 1
            scores[i] = float(min(1.0, best_score))

        return labels, scores


class MatchThreeAgent:
    ROW_NUM = 8
    COL_NUM = 8

    def __init__(self, top_left, bottom_right, elem_images, action_delay=0.5):
        self.prev_elem_array = None
        self.elem_array = np.zeros((8, 8), dtype=np.int64)
        self._grid_height = None
        self._grid_width = None
        self._grid_location = {}
        self._top_left = top_left
        self._bottom_right = bottom_right
        self._scores = [1] * self.ROW_NUM * self.COL_NUM
        self._action_delay = action_delay
        self._lookahead_discount = 0.72
        self._rollout_samples = 8
        self._rollout_topk = 4
        self._root_evaluation_cap = 6
        # self._elem_images = elem_images
        self._element_matcher = ElementMatcher(elem_images)

    def identify_game_board(self):
        x, y, w, h = self._top_left[0], self._top_left[1], self._bottom_right[0], self._bottom_right[1]
        board = pyautogui.screenshot(region=(x, y, w - x, h - y))
        board_array = get_image_array(board)
        return board_array

    def refresh_board_state(self):
        board_array = self.identify_game_board()
        grids = self.split_board_into_grids(board_array)
        self.update_elements(grids)
        return self.get_confidence_score()

    def should_take_action(self, wait_static):
        board_static = (self.prev_elem_array is None or np.array_equal(self.prev_elem_array, self.elem_array))
        return (not wait_static) or board_static

    @staticmethod
    def normalize_action(action):
        if not action:
            return None
        return (
            (int(action[0][0]), int(action[0][1])),
            (int(action[1][0]), int(action[1][1]))
        )

    def wait_until_board_settled(self, timeout=2.0, poll_interval=0.06, stable_frames=3, diff_threshold=2.0):
        start_ts = time.time()
        prev_gray = None
        stable_count = 0

        while time.time() - start_ts < timeout:
            board = self.identify_game_board()
            # Downscale to reduce computation and make settle detection cheaper.
            small_board = cv2.resize(board, (64, 64), interpolation=cv2.INTER_AREA)
            cur_gray = cv2.cvtColor(small_board, cv2.COLOR_BGR2GRAY)

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
        if self._grid_height is None:
            self._grid_height = height // self.ROW_NUM
            self._grid_width = width // self.COL_NUM
            self._element_matcher.set_grid_size(self._grid_width, self._grid_height)
            for i in range(self.ROW_NUM):
                for j in range(self.COL_NUM):
                    self._grid_location[(i, j)] = (self._top_left[0] + j * self._grid_width + self._grid_width / 2,
                                                   self._top_left[1] + i * self._grid_height + self._grid_height / 2)
            # for elem_name, image in self._elem_images.items():
            #     self._elem_images[elem_name] = cv2.resize(image, (self._grid_width, self._grid_height))  # unused

        grids = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                x_margin, y_margin = 0, 0
                start_x, end_x = int(j * self._grid_width + x_margin), int((j + 1) * self._grid_width - x_margin)
                start_y, end_y = int(i * self._grid_height + y_margin), int((i + 1) * self._grid_height - y_margin)
                grid = grid_array[start_y:end_y, start_x:end_x]
                grids.append(grid)

        return grids

    def update_elements(self, grids):
        labels, scores = self._element_matcher.classify_grids(grids)

        elem_array = np.zeros((8, 8), dtype=np.int64)

        self._scores = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            row, col = i // 8, i % 8
            elem_array[row, col] = int(label) + 1
            self._scores.append(float(score))

        self.prev_elem_array = self.elem_array
        self.elem_array = elem_array

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

        cur_grid_lst = [(i, j) for i in range(0, self.ROW_NUM) for j in range(0, self.COL_NUM)]
        ar_swap = np.zeros((self.ROW_NUM, self.COL_NUM))
        ar_match = np.zeros((self.ROW_NUM, self.COL_NUM))
        for index in cur_grid_lst:
            ar_swap[index] = self.elem_array[index]
            ar_match[index] = self.elem_array[index]

        action_set = set()
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-1] == ar_match[:, 1:], ar_match[:, :-1] != 0)):
            if index_could_swap((i, j - 1)):
                for index in [(i - 1, j - 1), (i + 1, j - 1), (i, j - 2)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i, j - 1), index))
            if index_could_swap((i, j + 2)):
                for index in [(i - 1, j + 2), (i + 1, j + 2), (i, j + 3)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i, j + 2), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:-1, :] == ar_match[1:, :], ar_match[:-1, :] != 0)):
            if index_could_swap((i - 1, j)):
                for index in [(i - 1, j - 1), (i - 1, j + 1), (i - 2, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i - 1, j), index))
            if index_could_swap((i + 2, j)):
                for index in [(i + 2, j - 1), (i + 2, j + 1), (i + 3, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i + 2, j), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-2] == ar_match[:, 2:], ar_match[:, :-2] != 0)):
            if index_could_swap((i, j + 1)):
                for index in [(i - 1, j + 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            action_set.add(((i, index[1]), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:-2, :] == ar_match[2:, :], ar_match[:-2, :] != 0)):
            if index_could_swap((i + 1, j)):
                for index in [(i + 1, j - 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            action_set.add(((index[0], j), index))

        return list(action_set)

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

        return score, simulated

    def _generate_valid_swap_actions(self, board):
        actions = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                for ni, nj in ((i + 1, j), (i, j + 1)):
                    if ni >= self.ROW_NUM or nj >= self.COL_NUM:
                        continue
                    swapped = board.copy()
                    swapped[(i, j)], swapped[(ni, nj)] = swapped[(ni, nj)], swapped[(i, j)]
                    if self._find_matches(swapped)["matched_cells"]:
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
        for sample_idx in range(self._rollout_samples):
            rng = random.Random(seed_base + sample_idx * 7919)
            cur_board = next_board.copy()
            sample_score = 0.0

            for d in range(depth - 1):
                actions = self._generate_valid_swap_actions(cur_board)
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

        # Enforce strict top-level priority:
        # 5-line / 3+3 cross > 4-line(horizontal) > 3-line.
        ranked_candidates = []
        for action in candidate_actions:
            simulated = board.copy()
            index1, index2 = action
            simulated[index1], simulated[index2] = simulated[index2], simulated[index1]
            match_info = self._find_matches(simulated)
            priority_type, _ = self._classify_match_priority(match_info)
            ranked_candidates.append((self._priority_rank(priority_type), action))

        max_rank = max(rank for rank, _ in ranked_candidates)
        candidate_actions = [action for rank, action in ranked_candidates if rank == max_rank]

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
                pyautogui.click(index1[0], index1[1])
                return
            pyautogui.moveTo(index1[0], index1[1])
            pyautogui.mouseDown()
            pyautogui.moveTo(index2[0], index2[1], duration=0.22)
            pyautogui.mouseUp()

        if not action:
            return []

        screen_index1, screen_index2 = self._grid_location[action[0]], self._grid_location[action[1]]
        swap_element(screen_index1, screen_index2)
        # Give the board a short settle time to avoid issuing the next move too early.
        time.sleep(self._action_delay)
        return action

    def get_confidence_score(self):
        return np.mean(self._scores)


def run(wait_static, show, action_delay, lookahead_depth, disable_wait_settle, settle_timeout):
    wait_for_start_signal()

    cur_path = sys.argv[0]
    cur_dir = os.path.dirname(cur_path)
    resource_dir = os.path.join(cur_dir, 'resource')

    top_left, bottom_right = resolve_board_coordinates(resource_dir=resource_dir)
    show_board_if_needed(show, top_left, bottom_right)

    elem_images = load_elem_images(resource_dir)
    agent = MatchThreeAgent(top_left, bottom_right, elem_images, action_delay=action_delay)
    last_board_signature = None
    last_action = None

    while True:
        should_stop, last_board_signature, last_action = run_main_cycle(
            agent=agent,
            wait_static=wait_static,
            lookahead_depth=lookahead_depth,
            disable_wait_settle=disable_wait_settle,
            settle_timeout=settle_timeout,
            last_board_signature=last_board_signature,
            last_action=last_action
        )
        if should_stop:
            break


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

    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument('--lookahead_depth', type=int, default=DEFAULT_LOOKAHEAD_DEPTH,
                          help=f'search depth for move planning, recommended 3~4 (default: {DEFAULT_LOOKAHEAD_DEPTH})')
    advanced.add_argument('--disable_wait_settle', action="store_true",
                          help='disable waiting for board to settle after each swap')
    advanced.add_argument('--settle_timeout', type=float, default=DEFAULT_SETTLE_TIMEOUT,
                          help=f'max seconds to wait for board settling after each swap (default: {DEFAULT_SETTLE_TIMEOUT})')
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
            settle_timeout=max(0.3, args.settle_timeout)
        )
    except Exception as e:
        print('An error occurred: {}'.format(e))
        time.sleep(5)
        raise
