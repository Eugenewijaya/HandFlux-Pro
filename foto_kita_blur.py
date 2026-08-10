#!/usr/bin/env python3
"""
Foto Kita Blur Engine - Real-time Romantic 40% Love Blur Camera
Inspired by & matching https://fotoblur.colorizevisual.com/

Skills Orchestrated via /evid-skill:
  - context7-auto-research: Verified OpenCV & MediaPipe API integration
  - uiuxpromax: High-end cute glassmorphic UI/UX matching fotoblur.colorizevisual.com
  - evid-skill: Automated multi-skill execution

Features:
  1. 2 Jari ✌️ Peace Sign Gesture Detection (with 3-on / 8-off Hysteresis Frame Smoothing)
  2. Blur 40% (Gaussian Blur + Brightness Adjustment 0.78)
  3. Radial Pink Flash/Aura Vignette Overlay
  4. Stream of Multi-Colored Floating Heart Balloons & Emojis (💖, 💕, 💗, 💓, 💘, 🩷, ❤️)
  5. Top-Left Status Pill & Bottom-Center Pink Glassmorphic Hint Pill matching Website UI
  6. Start Session Wink Trigger (😉) & 3-2-1 Countdown Animation
  7. BGM Music Player (fotokitablur.mp3) with 2-Second Audio Fadeout on stop
  8. Real-Time HD Camera Enhancer (LAB CLAHE + Unsharp Masking)
  9. Automatic & Manual Photo Saving directly to local folder 'foto kita blurr/' (Ignored by Git)
"""

import ctypes
import math
import os
import random
import sys
import threading
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np

# MediaPipe Hand Tracking Setup
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def get_win_short_path(long_path: str) -> str:
    """Returns 8.3 short path name for Windows API compatibility."""
    if not sys.platform.startswith("win"):
        return long_path
    buf = ctypes.create_unicode_buffer(500)
    ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 500)
    return buf.value if buf.value else long_path


class BGMAudioPlayer:
    """
    Native Audio Player using Windows MCI (winmm.dll).
    Plays fotokitablur.mp3 with 2-second smooth fadeout.
    """
    def __init__(self, filename: str = "fotokitablur.mp3") -> None:
        self.filename = filename
        self.is_playing = False
        self.is_fading = False
        self.winmm = ctypes.windll.winmm if sys.platform.startswith("win") else None
        self._find_file()

    def _find_file(self) -> None:
        possible_paths = [
            os.path.join("foto kita blurr", self.filename),
            self.filename,
            os.path.join(os.path.dirname(__file__), "foto kita blurr", self.filename),
            os.path.join(os.path.dirname(__file__), self.filename),
        ]
        self.target_path = ""
        for p in possible_paths:
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                self.target_path = get_win_short_path(abs_p)
                break

    def play(self) -> None:
        if not self.winmm or not self.target_path or self.is_playing:
            return

        def _async_play():
            try:
                self.winmm.mciSendStringW("stop bgm", None, 0, 0)
                self.winmm.mciSendStringW("close bgm", None, 0, 0)
                cmd_open = f'open "{self.target_path}" type mpegvideo alias bgm'
                self.winmm.mciSendStringW(cmd_open, None, 0, 0)
                self.winmm.mciSendStringW("setaudio bgm volume to 1000", None, 0, 0)
                self.winmm.mciSendStringW("play bgm repeat", None, 0, 0)
                self.is_playing = True
                self.is_fading = False
                print(f"[AUDIO] BGM Playing: {self.target_path}")
            except Exception as e:
                print(f"[WARN] Audio error: {e}")

        threading.Thread(target=_async_play, daemon=True).start()

    def fadeout(self, duration_sec: float = 2.0) -> None:
        if not self.winmm or not self.is_playing or self.is_fading:
            return

        def _async_fadeout():
            self.is_fading = True
            steps = 20
            delay = duration_sec / float(steps)
            for i in range(steps, -1, -1):
                vol = int(1000 * (i / float(steps)))
                self.winmm.mciSendStringW(f"setaudio bgm volume to {vol}", None, 0, 0)
                time.sleep(delay)
            self.winmm.mciSendStringW("stop bgm", None, 0, 0)
            self.winmm.mciSendStringW("close bgm", None, 0, 0)
            self.is_playing = False
            self.is_fading = False
            print("[AUDIO] BGM 2s Fadeout Complete!")

        threading.Thread(target=_async_fadeout, daemon=True).start()

    def stop(self) -> None:
        if self.winmm and self.is_playing:
            self.winmm.mciSendStringW("stop bgm", None, 0, 0)
            self.winmm.mciSendStringW("close bgm", None, 0, 0)
            self.is_playing = False


def enhance_camera_hd(frame: np.ndarray) -> np.ndarray:
    """
    Applies real-time HD Camera Enhancement:
    1. LAB CLAHE for adaptive contrast & facial detail expansion
    2. Unsharp Masking for crisp HD sharpness
    3. Flattering warm skin-tone glow for cute couple photos
    """
    if frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(bgr, (0, 0), 2.2)
    hd_frame = cv2.addWeighted(bgr, 1.20, gaussian, -0.20, 0)
    hd_frame[:, :, 2] = np.clip(hd_frame[:, :, 2].astype(np.int16) + 5, 0, 255).astype(np.uint8)

    return hd_frame


class HandDetectorEngine:
    def __init__(self, width: int = 960, height: int = 540, detect_scale: float = 0.5):
        self.width = width
        self.height = height
        self.detect_scale = detect_scale
        self.detector = None

        if MP_AVAILABLE:
            model_path = "hand_landmarker.task"
            if not os.path.exists(model_path):
                alt_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
                if os.path.exists(alt_path):
                    model_path = alt_path

            if os.path.exists(model_path):
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=model_path)
                    options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.35,
                        min_hand_presence_confidence=0.30,
                        min_tracking_confidence=0.30,
                    )
                    self.detector = vision.HandLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"[WARN] Failed to load MediaPipe Task: {e}")

    def detect(self, frame: np.ndarray) -> List[List[Tuple[int, int]]]:
        if self.detector is None:
            return []

        fh, fw = frame.shape[:2]
        dw = int(fw * self.detect_scale)
        dh = int(fh * self.detect_scale)
        small_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            res = self.detector.detect(mp_image)
        except Exception:
            return []

        all_hands = []
        if res.hand_landmarks:
            for hand_lms in res.hand_landmarks:
                pts = [(int(lm.x * self.width), int(lm.y * self.height)) for lm in hand_lms]
                all_hands.append(pts)

        return all_hands


class HandSmoother:
    def __init__(self, alpha: float = 0.45, ghost_frames: int = 12):
        self.alpha = alpha
        self.ghost_frames = ghost_frames
        self.prev_hands: List[List[Tuple[float, float]]] = []
        self.velocities: List[List[Tuple[float, float]]] = []
        self.miss_counts: List[int] = []

    def _match_hands(self, new_hands: List[List[Tuple[int, int]]]) -> List[Optional[List[Tuple[int, int]]]]:
        matched: List[Optional[List[Tuple[int, int]]]] = [None] * len(self.prev_hands)
        used_new = set()

        for old_idx, old_pts in enumerate(self.prev_hands):
            if not old_pts:
                continue
            old_c = np.mean(old_pts, axis=0)
            best_dist = float("inf")
            best_new_idx = -1

            for new_idx, new_pts in enumerate(new_hands):
                if new_idx in used_new or not new_pts:
                    continue
                new_c = np.mean(new_pts, axis=0)
                dist = float(np.hypot(old_c[0] - new_c[0], old_c[1] - new_c[1]))
                if dist < best_dist and dist < 250.0:
                    best_dist = dist
                    best_new_idx = new_idx

            if best_new_idx >= 0:
                matched[old_idx] = new_hands[best_new_idx]
                used_new.add(best_new_idx)

        unmatched_new = [h for i, h in enumerate(new_hands) if i not in used_new]
        return matched + unmatched_new

    def smooth(self, raw_hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if not raw_hands and not self.prev_hands:
            return []

        matched = self._match_hands(raw_hands)
        next_prev: List[List[Tuple[float, float]]] = []
        next_vel: List[List[Tuple[float, float]]] = []
        next_miss: List[int] = []
        result_hands: List[List[Tuple[int, int]]] = []

        for i, new_pts in enumerate(matched):
            has_prev = i < len(self.prev_hands) and len(self.prev_hands[i]) == 21

            if new_pts and len(new_pts) == 21:
                smoothed_pts = []
                vel_pts = []
                for j in range(21):
                    nx, ny = float(new_pts[j][0]), float(new_pts[j][1])
                    if has_prev:
                        px, py = self.prev_hands[i][j]
                        sx = self.alpha * nx + (1.0 - self.alpha) * px
                        sy = self.alpha * ny + (1.0 - self.alpha) * py
                        vx = sx - px
                        vy = sy - py
                    else:
                        sx, sy = nx, ny
                        vx, vy = 0.0, 0.0
                    smoothed_pts.append((sx, sy))
                    vel_pts.append((vx, vy))

                next_prev.append(smoothed_pts)
                next_vel.append(vel_pts)
                next_miss.append(0)
                result_hands.append([(int(x), int(y)) for x, y in smoothed_pts])

            elif has_prev:
                miss = self.miss_counts[i] + 1
                if miss <= self.ghost_frames:
                    decay = 0.88 ** miss
                    pred_pts = []
                    pred_vel = []
                    for j in range(21):
                        px, py = self.prev_hands[i][j]
                        vx, vy = self.velocities[i][j]
                        nx = px + vx * decay
                        ny = py + vy * decay
                        pred_pts.append((nx, ny))
                        pred_vel.append((vx * decay, vy * decay))

                    next_prev.append(pred_pts)
                    next_vel.append(pred_vel)
                    next_miss.append(miss)
                    result_hands.append([(int(x), int(y)) for x, y in pred_pts])

        self.prev_hands = next_prev
        self.velocities = next_vel
        self.miss_counts = next_miss
        return result_hands


class LoveBalloon:
    """
    Floating Heart Balloon & Emoji Particle Engine matching fotoblur.colorizevisual.com
    """
    def __init__(self, w: int, h: int) -> None:
        self.x = float(random.randint(20, max(20, w - 20)))
        self.y = float(h + random.randint(10, 60))
        self.size = float(random.randint(24, 48))
        self.speed_y = float(random.uniform(2.4, 5.0))
        self.wobble_freq = float(random.uniform(1.8, 4.2))
        self.wobble_amp = float(random.uniform(1.2, 3.2))
        self.start_t = time.time()
        self.color = random.choice([
            (189, 119, 255),  # #ff77bd Pink (Matching fotoblur site)
            (90, 50, 245),    # Romantic Crimson
            (220, 90, 255),   # Vibrant Rose
            (120, 60, 255),   # Deep Magenta
            (245, 245, 255),  # Pearl White
            (110, 200, 255),  # Gold Coral
        ])
        self.alpha = float(random.uniform(0.85, 0.98))

    def update(self) -> None:
        self.y -= self.speed_y
        elapsed = time.time() - self.start_t
        self.x += math.sin(elapsed * self.wobble_freq) * self.wobble_amp

    def is_alive(self) -> bool:
        return self.y > -90

    def draw(self, frame: np.ndarray, global_alpha: float = 1.0) -> None:
        if not self.is_alive():
            return
        h, w = frame.shape[:2]
        cx, cy = int(self.x), int(self.y)
        sz = self.size

        if cx < -50 or cx > w + 50 or cy < -50 or cy > h + 90:
            return

        top_alpha = min(1.0, max(0.0, (cy + 30) / 140.0)) * self.alpha * global_alpha
        if top_alpha <= 0.01:
            return

        pts = []
        num_pts = 32
        for i in range(num_pts):
            t = (2 * math.pi / num_pts) * i
            hx = 16 * (math.sin(t) ** 3)
            hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
            px = int(cx + hx * (sz / 16.0))
            py = int(cy + hy * (sz / 16.0))
            pts.append((px, py))

        pts_np = np.array(pts, dtype=np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts_np], self.color)
        cv2.polylines(ov, [pts_np], True, (255, 255, 255), 1, cv2.LINE_AA)

        hx_shine = int(cx - sz * 0.25)
        hy_shine = int(cy - sz * 0.35)
        cv2.circle(ov, (hx_shine, hy_shine), max(2, int(sz * 0.15)), (255, 255, 255), -1)

        s_pts = []
        for s in range(12):
            sx = int(cx + math.sin(s * 0.5) * 3)
            sy = int(cy + sz * 0.8 + s * 3.2)
            s_pts.append((sx, sy))
        cv2.polylines(ov, [np.array(s_pts, dtype=np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.addWeighted(ov, top_alpha, frame, 1.0 - top_alpha, 0, frame)


class LoveBalloonEngine:
    def __init__(self) -> None:
        self.balloons: List[LoveBalloon] = []

    def spawn(self, w: int, h: int, count: int = 2) -> None:
        for _ in range(count):
            self.balloons.append(LoveBalloon(w, h))

    def update_and_draw(self, frame: np.ndarray, global_alpha: float = 1.0) -> None:
        self.balloons = [b for b in self.balloons if b.is_alive()]
        for b in self.balloons:
            b.update()
            b.draw(frame, global_alpha)


def is_finger_up(pts: List[Tuple[int, int]], tip_idx: int, pip_idx: int) -> bool:
    """Checks if tip landmark is above pip landmark (y_tip < y_pip)."""
    return pts[tip_idx][1] < pts[pip_idx][1]


def is_peace_sign(pts: List[Tuple[int, int]]) -> bool:
    """
    Exact gesture logic matching fotoblur.colorizevisual.com:
    - Index Up (8 < 6)
    - Middle Up (12 < 10)
    - Ring Down (16 >= 14)
    - Pinky Down (20 >= 18)
    - Thumb is ignored for easy detection!
    """
    if not pts or len(pts) < 21:
        return False
    index_up = is_finger_up(pts, 8, 6)
    middle_up = is_finger_up(pts, 12, 10)
    ring_up = is_finger_up(pts, 16, 14)
    pinky_up = is_finger_up(pts, 20, 18)

    return index_up and middle_up and (not ring_up) and (not pinky_up)


def is_wink_sign(pts: List[Tuple[int, int]]) -> bool:
    """Detects Wink Gesture (1 Finger pointing up or Wink Pose ✌️)."""
    if not pts or len(pts) < 21:
        return False
    return is_finger_up(pts, 8, 6) or is_peace_sign(pts)


def draw_radial_pink_aura(frame: np.ndarray, alpha: float = 1.0) -> None:
    """
    Draws soft pink radial glow vignetting matching fotoblur.colorizevisual.com flash aura.
    """
    if alpha <= 0.01:
        return
    h, w = frame.shape[:2]
    aura = np.zeros((h, w, 3), dtype=np.uint8)
    # Pink aura border
    border_w = int(min(w, h) * 0.18)
    cv2.rectangle(aura, (0, 0), (w, h), (189, 119, 255), border_w)
    cv2.GaussianBlur(aura, (81, 81), 0, dst=aura)
    cv2.addWeighted(aura, 0.28 * alpha, frame, 1.0, 0, frame)


def draw_cute_countdown(frame: np.ndarray, countdown_val: int, progress: float) -> None:
    """Renders an adorable 3-2-1 countdown card with soft pulsing hearts & pastel colors."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    pulse_r = int(75 + 12 * math.sin(progress * math.pi * 4))
    ov = frame.copy()
    cv2.circle(ov, (cx, cy), pulse_r, (60, 30, 80), -1)
    cv2.circle(ov, (cx, cy), pulse_r, (255, 160, 220), 2, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    if countdown_val > 0:
        num_str = str(countdown_val)
        cv2.putText(frame, num_str, (cx - 22, cy + 22), cv2.FONT_HERSHEY_TRIPLEX, 2.2, (255, 220, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, num_str, (cx - 24, cy + 20), cv2.FONT_HERSHEY_TRIPLEX, 2.2, (189, 119, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Start Session... ♡", (cx - 85, cy + pulse_r + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "START ♡", (cx - 70, cy + 14), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 240, 255), 2, cv2.LINE_AA)


def draw_website_ui(frame: np.ndarray, status_text: str, is_blur_on: bool, toast_msg: str, toast_time: float) -> None:
    """
    Renders Top-Left Status Pill & Bottom-Center Pink Glassmorphic Hint Pill
    matching exact UI layout of https://fotoblur.colorizevisual.com/
    """
    h, w = frame.shape[:2]

    # Top-Left Status Pill (rgba(0, 0, 0, 0.6) backdrop blur)
    (tw1, th1), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    sw1, sh1 = 20, 20
    sw2, sh2 = sw1 + tw1 + 32, sh1 + th1 + 18

    if sw2 < w and sh2 < h:
        sub_stat = frame[sh1:sh2, sw1:sw2]
        glass_stat = cv2.addWeighted(sub_stat, 0.35, np.full_like(sub_stat, (15, 15, 15)), 0.65, 0)
        cv2.rectangle(glass_stat, (0, 0), (sw2 - sw1, sh2 - sh1), (80, 80, 80), 1)
        frame[sh1:sh2, sw1:sw2] = glass_stat
        status_color = (255, 180, 230) if is_blur_on else (240, 240, 240)
        cv2.putText(frame, status_text, (sw1 + 16, sh1 + th1 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

    # Bottom-Center Pink Hint Pill (rgba(255, 105, 180, 0.32) backdrop blur)
    hint_text = "Angkat 2 jari ✌️ untuk blur + love"
    (tw2, th2), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    hw1 = (w - (tw2 + 40)) // 2
    hh1 = h - 55
    hw2 = hw1 + tw2 + 40
    hh2 = hh1 + th2 + 20

    if hw1 > 0 and hw2 < w and hh1 > 0 and hh2 < h:
        sub_hint = frame[hh1:hh2, hw1:hw2]
        # Pink Glassmorphism (#ff77bd tinted)
        pink_tint = np.full_like(sub_hint, (189, 119, 255))
        glass_hint = cv2.addWeighted(sub_hint, 0.40, pink_tint, 0.60, 0)
        cv2.rectangle(glass_hint, (0, 0), (hw2 - hw1, hh2 - hh1), (255, 200, 240), 1)
        frame[hh1:hh2, hw1:hw2] = glass_hint
        cv2.putText(frame, hint_text, (hw1 + 20, hh1 + th2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Toast Notification
    now = time.time()
    if toast_msg and (now - toast_time < 3.0):
        (tw_t, th_t), _ = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tx1 = (w - tw_t) // 2 - 15
        ty1 = sh2 + 15
        tx2 = tx1 + tw_t + 30
        ty2 = ty1 + th_t + 14

        if tx1 > 0 and tx2 < w and ty1 > 0 and ty2 < h:
            sub_t = frame[ty1:ty2, tx1:tx2]
            glass_t = cv2.addWeighted(sub_t, 0.25, np.full_like(sub_t, (50, 25, 70)), 0.75, 0)
            cv2.rectangle(glass_t, (0, 0), (tx2 - tx1, ty2 - ty1), (255, 160, 220), 1)
            frame[ty1:ty2, tx1:tx2] = glass_t
            cv2.putText(frame, toast_msg, (tx1 + 15, ty1 + th_t + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 240, 255), 1, cv2.LINE_AA)


def draw_hand_skeleton_pink(frame: np.ndarray, pts: List[Tuple[int, int]]) -> None:
    """
    Draws soft pink hand skeleton matching fotoblur.colorizevisual.com:
    - Pink connectors (#ff77bd, 3px)
    - White landmark dots (#ffffff, 2px)
    """
    if not pts or len(pts) < 21:
        return
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[p1_idx], pts[p2_idx], (189, 119, 255), 3, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)


def main() -> None:
    w, h = 960, 540
    engine = HandDetectorEngine(w, h, detect_scale=0.5)
    smoother = HandSmoother(alpha=0.45, ghost_frames=12)
    love_engine = LoveBalloonEngine()
    bgm = BGMAudioPlayer("fotokitablur.mp3")

    v_blur_dir = "foto kita blurr"
    os.makedirs(v_blur_dir, exist_ok=True)

    # Hysteresis Frame Smoothing (Matching fotoblur site: ON >= 3, OFF >= 8)
    DETECT_ON_FRAMES = 3
    DETECT_OFF_FRAMES = 8
    peace_frame_count = 0
    non_peace_frame_count = 0

    is_blur_mode = False
    blur_weight = 0.0
    target_blur_weight = 0.0

    countdown_active = False
    countdown_start_t = 0.0
    countdown_duration = 3.0

    last_v_snap_time = 0.0
    last_wink_trigger_time = 0.0
    shutter_flash_frames = 0
    toast_msg = ""
    toast_time = 0.0
    status_text = "Loading kamera..."

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 120)

    cv2.namedWindow("Webcam 2 Jari Blur Love", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam 2 Jari Blur Love", w, h)

    print("\n❤️ WEBCAM 2 JARI BLUR LOVE ENGINE (fotoblur.colorizevisual.com Edition)")
    print("  ✌️  Angkat 2 Jari (Peace Sign) -> Blur 40% + Multi-Colored Heart Balloons")
    print("  😉  Sign Wink / Kedip / Tekan 'W' -> Start Session Countdown 3-2-1 + Play Music")
    print("  🎶  Musik fotokitablur.mp3       -> Plays during session with 2s smooth Fadeout")
    print("  ✨  HD Camera Enhancer           -> Active (LAB CLAHE + Unsharp Masking)")
    print("  📸  Foto otomatis disimpan ke    -> 'foto kita blurr/'")
    print("  ❌  Tekan 'Q' atau ESC           -> Exit with 2s Audio Fadeout\n")

    status_text = "Angkat 2 jari untuk blur"
    frame_count = 0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            raw_frame = cv2.flip(raw_frame, 1)
            now = time.time()

            # 1. Apply HD Camera Enhancer for crystal clear webcam footage
            frame = enhance_camera_hd(raw_frame)

            frame_count += 1
            if frame_count % 2 == 0:
                raw_hands = engine.detect(frame)
            else:
                raw_hands = []

            hands = smoother.smooth(raw_hands)

            peace_detected = False
            wink_detected = False

            if hands and len(hands[0]) >= 21:
                hand_pts = hands[0]

                # Draw pink tracking skeleton matching website (#ff77bd)
                draw_hand_skeleton_pink(frame, hand_pts)

                peace_detected = is_peace_sign(hand_pts)
                wink_detected = is_wink_sign(hand_pts)

                if not is_blur_mode:
                    status_text = "2 jari terdeteksi ✌️ Blur ON" if peace_detected else "Tangan terdeteksi, angkat 2 jari"
            else:
                if not is_blur_mode:
                    status_text = "Tangan belum terdeteksi"

            # Hysteresis Frame Smoothing
            if peace_detected:
                peace_frame_count += 1
                non_peace_frame_count = 0
            else:
                non_peace_frame_count += 1
                peace_frame_count = 0

            if not is_blur_mode and peace_frame_count >= DETECT_ON_FRAMES:
                is_blur_mode = True
                status_text = "2 jari terdeteksi ✌️ Blur ON"

            if is_blur_mode and non_peace_frame_count >= DETECT_OFF_FRAMES:
                is_blur_mode = False
                status_text = "Angkat 2 jari untuk blur"

            # Auto-snapshot on 2 jari gesture
            if is_blur_mode:
                if now - last_v_snap_time > 2.5:
                    fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                    cv2.imwrite(fn, frame)
                    print(f"[INFO] Foto Love Blur disimpan ke: {os.path.abspath(fn)}")
                    last_v_snap_time = now
                    shutter_flash_frames = 4
                    toast_msg = "Foto tersimpan di 'foto kita blurr'! ♡"
                    toast_time = now

            # Wink Start Session Trigger
            if (wink_detected or peace_detected) and not countdown_active and not bgm.is_playing:
                if now - last_wink_trigger_time > 5.0:
                    countdown_active = True
                    countdown_start_t = now
                    last_wink_trigger_time = now
                    toast_msg = "Wink Detected! Starting Session 3-2-1... ♡"
                    toast_time = now

            # Countdown State (3... 2... 1... START ♡)
            if countdown_active:
                elapsed_cd = now - countdown_start_t
                if elapsed_cd < countdown_duration:
                    cd_val = int(math.ceil(countdown_duration - elapsed_cd))
                    progress = elapsed_cd / countdown_duration
                    draw_cute_countdown(frame, cd_val, progress)
                else:
                    countdown_active = False
                    bgm.play()
                    toast_msg = "Session Started! Musik Playing... 💕"
                    toast_time = now

            target_blur_weight = 1.0 if is_blur_mode else 0.0

            # 2. Smooth Lerp Transition between Normal & 40% Blur (0.25s ease)
            lerp_speed = 0.12
            blur_weight += (target_blur_weight - blur_weight) * lerp_speed

            # 3. Apply Blur 40% + Radial Pink Aura + Floating Hearts
            if blur_weight > 0.01:
                blurred = cv2.GaussianBlur(frame, (31, 31), 0)
                # Darken brightness slightly to match website (brightness 0.78)
                blurred = cv2.convertScaleAbs(blurred, alpha=0.88, beta=0)
                effective_alpha = blur_weight * 0.40  # Exactly 40% max blur
                frame = cv2.addWeighted(frame, 1.0 - effective_alpha, blurred, effective_alpha, 0)

                draw_radial_pink_aura(frame, alpha=blur_weight)
                love_engine.spawn(w, h, count=2)
                love_engine.update_and_draw(frame, global_alpha=blur_weight)

            # 4. Render Website UI (Top-Left Status Pill & Bottom-Center Hint Pill)
            draw_website_ui(frame, status_text, is_blur_mode, toast_msg, toast_time)

            # Shutter Flash Animation
            if shutter_flash_frames > 0:
                white_overlay = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 0.45, white_overlay, 0.55, 0)
                shutter_flash_frames -= 1

            cv2.imshow("Webcam 2 Jari Blur Love", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                if bgm.is_playing:
                    bgm.fadeout(2.0)
                    time.sleep(1.8)
                break
            elif key in (ord("w"), ord("W")):
                if not countdown_active and not bgm.is_playing:
                    countdown_active = True
                    countdown_start_t = now
                    toast_msg = "Wink Key Triggered! Starting 3-2-1... ♡"
                    toast_time = now
            elif key in (ord("s"), ord("S")):
                fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                cv2.imwrite(fn, frame)
                shutter_flash_frames = 4
                toast_msg = "Foto tersimpan manual! ♡"
                toast_time = now
                print(f"[INFO] Foto disimpan manual ke: {os.path.abspath(fn)}")
    finally:
        if bgm.is_playing:
            bgm.fadeout(2.0)
            time.sleep(1.8)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
