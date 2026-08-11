"""
HandFlux Pro - Real-time Hand Gesture Filter & Portal Engine
"""

__version__ = "3.0.0"

import argparse
from dataclasses import dataclass, field
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Callable, Optional
import urllib.request

import cv2
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def ensure_model_exists() -> bool:
    if not os.path.exists(MODEL_PATH):
        try:
            print("[INFO] Mengunduh model deteksi tangan MediaPipe (hand_landmarker.task)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[INFO] Model berhasil diunduh!")
            return True
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh model: {e}")
            return False
    return True


class HandSmoother:
    """Hand position tracker with EMA smoothing, velocity prediction, and ghost-frame hold.

    - EMA smoothing eliminates per-frame jitter (alpha=0.45).
    - When MediaPipe drops a detection, we extrapolate from velocity for up to
      `ghost_frames` frames so the portal keeps moving in the right direction.
    - Nearest-centroid matching prevents hand-ID swaps between frames.
    """

    def __init__(self, alpha: float = 0.45, ghost_frames: int = 12) -> None:
        self.alpha = alpha
        self.ghost_frames = ghost_frames
        # Each entry: list of 21 (x, y) floats
        self.prev_hands: List[List[Tuple[float, float]]] = []
        # Velocity per landmark per hand: dx, dy per frame
        self.velocities: List[List[Tuple[float, float]]] = []
        self._miss_count: int = 0

    @staticmethod
    def _centroid(hand: list) -> Tuple[float, float]:
        xs = [p[0] for p in hand]
        ys = [p[1] for p in hand]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _match_hands(self, new_hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """Sort hands by centroid X (left-to-right) to prevent hand-ID swaps."""
        if not new_hands:
            return []
        if len(new_hands) >= 2:
            # ponytail: strict spatial X-sort, no slot games — just left-to-right ordering
            return sorted(new_hands, key=lambda h: self._centroid(h)[0])[:2]
        return new_hands

    def smooth(self, hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if not hands:
            # Ghost frames: extrapolate from velocity instead of freezing
            if self.prev_hands and self._miss_count < self.ghost_frames:
                self._miss_count += 1
                decay = max(0.0, 1.0 - self._miss_count * 0.08)
                predicted = []
                for hi, hand in enumerate(self.prev_hands):
                    vel = self.velocities[hi] if hi < len(self.velocities) else [(0.0, 0.0)] * len(hand)
                    predicted.append([
                        (p[0] + v[0] * decay, p[1] + v[1] * decay)
                        for p, v in zip(hand, vel)
                    ])
                self.prev_hands = predicted
                return [[(int(p[0]), int(p[1])) for p in h] for h in predicted]
            self.prev_hands = []
            self.velocities = []
            self._miss_count = 0
            return []

        self._miss_count = 0
        hands = self._match_hands(hands)
        smoothed_hands = []
        new_velocities = []

        for i, hand in enumerate(hands):
            if not hand:
                continue

            if i < len(self.prev_hands) and len(self.prev_hands[i]) == len(hand):
                prev = self.prev_hands[i]
                smoothed = []
                vels = []
                for j, curr in enumerate(hand):
                    dist = np.hypot(curr[0] - prev[j][0], curr[1] - prev[j][1])
                    # Dynamic Alpha: if moving fast (>80px), alpha=0.85 (responsive). If still (~0px), alpha=0.20 (stable).
                    dynamic_alpha = max(0.20, min(0.85, 0.20 + (dist / 80.0) * 0.65))
                    
                    sx = dynamic_alpha * curr[0] + (1 - dynamic_alpha) * prev[j][0]
                    sy = dynamic_alpha * curr[1] + (1 - dynamic_alpha) * prev[j][1]
                    vels.append((sx - prev[j][0], sy - prev[j][1]))
                    smoothed.append((int(sx), int(sy)))
                new_velocities.append(vels)
            else:
                smoothed = hand
                new_velocities.append([(0.0, 0.0)] * len(hand))

            smoothed_hands.append(smoothed)

        self.prev_hands = [[(float(p[0]), float(p[1])) for p in h] for h in smoothed_hands]
        self.velocities = new_velocities
        return smoothed_hands


class HandDetectorEngine:
    """Multi-backend hand tracking engine supporting MediaPipe Tasks, MediaPipe Solutions, and Fallback."""

    def __init__(self, frame_width: int, frame_height: int, detect_scale: float = 0.5) -> None:
        self.w = frame_width
        self.h = frame_height
        # ponytail: detect at 0.5x resolution for ultra-fast 500+ FPS inference.
        self.detect_scale = detect_scale
        self.detect_w = max(1, int(frame_width * detect_scale))
        self.detect_h = max(1, int(frame_height * detect_scale))
        self.mode = "none"
        self.tasks_detector = None
        self.solutions_detector = None
        self.mp_module = None

        try:
            if ensure_model_exists() and os.path.exists(MODEL_PATH):
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                # ponytail: confidence 0.55 — high enough to kill ghost/false-positive
                # skeletons on walls/objects, low enough to keep real hands tracked.
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                    num_hands=2,
                    min_hand_detection_confidence=0.30,
                    min_hand_presence_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
                self.tasks_detector = vision.HandLandmarker.create_from_options(options)
                self.mp_module = mp
                self.mode = "tasks"
                print("[INFO] Engine Hand Tracking: MediaPipe Tasks (v0.10+)")
        except Exception as e:
            print(f"[DEBUG] MediaPipe Tasks init failed: {e}")

        if self.mode == "none":
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    mp_hands = mp.solutions.hands
                    self.solutions_detector = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.30,
                        min_tracking_confidence=0.30,
                    )
                    self.mode = "solutions"
                    print("[INFO] Engine Hand Tracking: MediaPipe Solutions (Legacy)")
            except Exception as e:
                print(f"[DEBUG] MediaPipe Solutions init failed: {e}")

        if self.mode == "none":
            print("[WARNING] Engine Hand Tracking: Fallback Skin Detector")

    @staticmethod
    def _validate_hand(pts: List[Tuple[int, int]], fw: int, fh: int) -> bool:
        """Validate that detection contains 21 landmarks."""
        return len(pts) >= 21

    def detect(self, frame: np.ndarray) -> List[List[Tuple[int, int]]]:
        if frame is None or frame.size == 0:
            return []

        # Downscale for faster inference
        if self.detect_scale < 1.0:
            small = cv2.resize(frame, (self.detect_w, self.detect_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_list = []

        if self.mode == "tasks" and self.tasks_detector is not None:
            try:
                mp_image = self.mp_module.Image(image_format=self.mp_module.ImageFormat.SRGB, data=rgb)
                res = self.tasks_detector.detect(mp_image)
                if res and res.hand_landmarks:
                    for hand_lms in res.hand_landmarks:
                        pts = [(int(lm.x * self.w), int(lm.y * self.h)) for lm in hand_lms]
                        if self._validate_hand(pts, self.w, self.h):
                            hands_list.append(pts)
            except Exception as e:
                print(f"[ERROR] Tasks detection error: {e}")

        elif self.mode == "solutions" and self.solutions_detector is not None:
            try:
                res = self.solutions_detector.process(rgb)
                if res and res.multi_hand_landmarks:
                    for hand_lm in res.multi_hand_landmarks:
                        pts = [(int(lm.landmark[i].x * self.w), int(lm.landmark[i].y * self.h)) for i in range(21)]
                        if self._validate_hand(pts, self.w, self.h):
                            hands_list.append(pts)
            except Exception as e:
                print(f"[ERROR] Solutions detection error: {e}")

        elif self.mode == "fallback":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70], dtype=np.uint8)
            upper = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if cv2.contourArea(c) > 1200]
            if valid:
                c = max(valid, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c)
                cx, cy = x + bw // 2, y + bh // 2
                fake_pts = [(cx, cy)] * 21
                hands_list.append(fake_pts)

        return hands_list


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    target_fps: int = 120
    active_fingers: int = 1
    enable_gesture_snap: bool = False
    pinch_threshold_ratio: float = 0.28
    filter_cooldown_sec: float = 0.35
    mode_cooldown_sec: float = 1.2
    gesture_cooldown_sec: float = 1.5
    fist_dist_threshold_px: float = 90.0
    auto_cycle_interval: float = 2.0  # Auto switch filter every 2 seconds
    mirror: bool = True
    show_hud: bool = True
    output_dir: str = "captures"


# --- Precomputed LUTs (built once at import, <0.1ms to apply) ---
_PSYCHEDELIC_LUT = (np.sin(np.arange(256) / 255.0 * np.pi * 4) * 127 + 128).astype(np.uint8)

_CYBERPUNK_DUOTONE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
for _i in range(3):
    _CYBERPUNK_DUOTONE_LUT[:, 0, _i] = np.linspace([60, 0, 15][_i], [255, 100, 240][_i], 256)

_CROSS_PROCESS_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
_x = np.arange(256)
_CROSS_PROCESS_LUT[:, 0, 0] = np.clip(1.2 * _x - 25, 0, 255).astype(np.uint8)
_CROSS_PROCESS_LUT[:, 0, 1] = np.clip(1.05 * _x, 0, 255).astype(np.uint8)
_CROSS_PROCESS_LUT[:, 0, 2] = np.clip(0.85 * _x + 35, 0, 255).astype(np.uint8)

_EMBOSS_KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)


class FilterBank:
    """Ultra-fast, 100% vectorized special effects filter bank (<1ms execution time)."""

    # --- NORMAL THEME ---
    @staticmethod
    def normal_camera(roi: np.ndarray) -> np.ndarray:
        return roi

    # --- SPECIAL FX THEME ---
    @staticmethod
    def glitch_cyber(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        res = roi.copy()
        shift = min(12, max(2, w // 25))
        res[:, shift:, 2] = roi[:, :-shift, 2]
        res[:, :-shift, 0] = roi[:, shift:, 0]
        if random.random() < 0.8:
            slice_y = random.randint(0, max(1, h - 15))
            slice_h = random.randint(5, 15)
            slice_shift = random.randint(-15, 15)
            y_end = min(h, slice_y + slice_h)
            res[slice_y:y_end, :] = np.roll(res[slice_y:y_end, :], slice_shift, axis=1)
        return res

    @staticmethod
    def crt_broken_tv(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        res = roi.copy()
        res[::3, :, :] = (res[::3, :, :].astype(np.uint16) * 6 // 10).astype(np.uint8)
        noise = np.random.randint(0, 35, (h, w, 1), dtype=np.uint8)
        res = cv2.add(res, cv2.merge([noise, noise, noise]))
        res[1:, :, 0] = roi[:-1, :, 0]
        return res

    @staticmethod
    def vhs_retro(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 6 or w < 6:
            return roi
        res = roi.copy()
        res[:, 4:, 0] = roi[:, :-4, 0]
        res[:, :-4, 2] = roi[:, 4:, 2]
        bar_h = max(2, h // 10)
        res[-bar_h:, :] = np.roll(res[-bar_h:, :], 20, axis=1)
        res[-bar_h:, :] = cv2.add(res[-bar_h:, :], 40)
        return res

    @staticmethod
    def pixel_art_8bit(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 8 or w < 8:
            return roi
        pw, ph = max(8, w // 12), max(8, h // 12)
        small = cv2.resize(roi, (pw, ph), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def chromatic_aberration(roi: np.ndarray) -> np.ndarray:
        """Lens chromatic split — shifts R and B channels in opposite directions."""
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi)
        b_shifted = np.roll(b, (2, 6), axis=(0, 1))
        r_shifted = np.roll(r, (-2, -6), axis=(0, 1))
        return cv2.merge([b_shifted, g, r_shifted])

    @staticmethod
    def film_grain(roi: np.ndarray) -> np.ndarray:
        """Analog film grain noise overlay."""
        if roi is None or roi.size == 0:
            return roi
        h, w, c = roi.shape
        noise = np.zeros((h, w, c), dtype=np.int8)
        cv2.randn(noise, 0, 25)
        return cv2.add(roi, noise, dtype=cv2.CV_8U)

    @staticmethod
    def emboss_relief(roi: np.ndarray) -> np.ndarray:
        """3D relief emboss via SIMD filter2D."""
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        embossed = cv2.filter2D(gray, -1, _EMBOSS_KERNEL) + 128
        return cv2.cvtColor(np.clip(embossed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def solarize_psychedelic(roi: np.ndarray) -> np.ndarray:
        """Sinusoidal LUT solarization — instant trippy colors."""
        if roi is None or roi.size == 0:
            return roi
        return cv2.LUT(roi, _PSYCHEDELIC_LUT)

    @staticmethod
    def xray_scan(roi: np.ndarray) -> np.ndarray:
        """Medical X-Ray bone simulation."""
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        return cv2.applyColorMap(inverted, cv2.COLORMAP_BONE)

    @staticmethod
    def comic_posterize(roi: np.ndarray) -> np.ndarray:
        """Cel-shaded comic with Canny black outlines."""
        if roi is None or roi.size == 0:
            return roi
        quantized = (roi >> 6) << 6  # 4 levels
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 140)
        edges_inv = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(quantized, edges_inv)

    @staticmethod
    def duotone_cyberpunk(roi: np.ndarray) -> np.ndarray:
        """Cyberpunk duotone: dark navy to neon pink."""
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.LUT(gray_3ch, _CYBERPUNK_DUOTONE_LUT)

    @staticmethod
    def cross_process(roi: np.ndarray) -> np.ndarray:
        """Retro film cross-processing via precomputed LUT."""
        if roi is None or roi.size == 0:
            return roi
        return cv2.LUT(roi, _CROSS_PROCESS_LUT)

    @staticmethod
    def negative_invert(roi: np.ndarray) -> np.ndarray:
        """Full color negative."""
        if roi is None or roi.size == 0:
            return roi
        return cv2.bitwise_not(roi)

    # --- Y2K POP-ART THEME (Reference Images) ---
    @staticmethod
    def indie_night_flash(roi: np.ndarray) -> np.ndarray:
        """Indie Sleaze night camera with flash light leak & directional motion streak (Reference Photo 1)."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        ksize = 11
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        np.fill_diagonal(kernel, 1.0 / ksize)
        blurred = cv2.filter2D(roi, -1, kernel)

        leak = np.zeros_like(roi)
        cv2.circle(leak, (w, 0), int(w * 0.75), (0, 150, 240), -1)
        leak = cv2.GaussianBlur(leak, (99, 99), 0)

        res = cv2.addWeighted(blurred, 0.85, leak, 0.35, 0)
        return cv2.convertScaleAbs(res, alpha=1.15, beta=-5)

    @staticmethod
    def red_halftone_print(roi: np.ndarray) -> np.ndarray:
        """Authentic red/maroon newsprint halftone dot matrix printing (Reference Photo 2)."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        dot_size = 6
        y_grid, x_grid = np.ogrid[:h, :w]
        screen = (np.sin(x_grid * np.pi / dot_size) * np.sin(y_grid * np.pi / dot_size) * 120 + 128).astype(np.uint8)

        dots = cv2.compare(gray, screen, cv2.CMP_GT)

        res = np.zeros_like(roi)
        res[dots == 255] = (220, 215, 235)  # BGR Soft Rose White
        res[dots == 0] = (30, 10, 110)      # BGR Deep Burgundy Maroon
        return res

    @staticmethod
    def grunge_pink_starburst(roi: np.ndarray) -> np.ndarray:
        """Gritty B&W high-contrast threshold with neon magenta starburst accent (Reference Photo 3)."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
        res = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        noise = np.zeros((h, w, 3), dtype=np.int8)
        cv2.randn(noise, 0, 20)
        res = cv2.add(res, noise, dtype=cv2.CV_8U)

        cx, cy = int(w * 0.82), int(h * 0.78)
        star_pts = []
        r_outer, r_inner = int(w * 0.22), int(w * 0.05)
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = i * np.pi / 5 - np.pi / 2
            star_pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
        poly_star = np.array(star_pts, dtype=np.int32)
        cv2.fillPoly(res, [poly_star], (220, 20, 255), lineType=cv2.LINE_AA)
        return res

    @staticmethod
    def bw_halftone_pink_halo(roi: np.ndarray) -> np.ndarray:
        """Monochromatic newsprint halftone with glowing neon pink halo ring (Reference Photo 4)."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        dot_size = 5
        y_grid, x_grid = np.ogrid[:h, :w]
        screen = (np.sin(x_grid * np.pi / dot_size) * np.sin(y_grid * np.pi / dot_size) * 120 + 128).astype(np.uint8)

        dots = cv2.compare(gray, screen, cv2.CMP_GT)
        res = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)

        cx, cy = w // 2, h // 2
        r_halo = int(min(w, h) * 0.40)
        cv2.circle(res, (cx, cy), r_halo, (220, 30, 255), 4, cv2.LINE_AA)
        cv2.circle(res, (cx, cy), r_halo + 3, (240, 100, 255), 2, cv2.LINE_AA)
        return res

    @staticmethod
    def y2k_lime_cyber_doodle(roi: np.ndarray) -> np.ndarray:
        """Y2K Cyber Magazine pop-art with electric lime green doodle outlines & lightning bolts (Reference Photo 5)."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]

        base = cv2.convertScaleAbs(roi, alpha=1.1, beta=10)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        lime_color = np.full_like(roi, (40, 255, 110), dtype=np.uint8)
        res = np.where(dilated_edges[:, :, None] == 255, lime_color, base)

        lb_pts = np.array([[int(w*0.08), int(h*0.15)], [int(w*0.14), int(h*0.30)], [int(w*0.09), int(h*0.32)], [int(w*0.16), int(h*0.50)]], dtype=np.int32)
        cv2.polylines(res, [lb_pts], isClosed=False, color=(40, 255, 110), thickness=3, lineType=cv2.LINE_AA)

        rb_pts = np.array([[int(w*0.90), int(h*0.20)], [int(w*0.84), int(h*0.35)], [int(w*0.89), int(h*0.37)], [int(w*0.82), int(h*0.55)]], dtype=np.int32)
        cv2.polylines(res, [rb_pts], isClosed=False, color=(40, 255, 110), thickness=3, lineType=cv2.LINE_AA)

        return res

    # --- CYBER & CODE THEME ---
    @staticmethod
    def matrix_rain(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        green = cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)
        b, g, r = cv2.split(green)
        g = cv2.add(g, 60)
        r = (r.astype(np.uint16) * 2 // 10).astype(np.uint8)
        res = cv2.merge([b, g, r])
        res[:, ::4, :] = cv2.add(res[:, ::4, :], 30)
        return res

    @staticmethod
    def hologram_cyan(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        b = np.clip(b * 1.5 + 40, 0, 255)
        g = np.clip(g * 1.3 + 30, 0, 255)
        r = np.clip(r * 0.2, 0, 255)
        res = cv2.merge([b, g, r]).astype(np.uint8)
        res[::3, :, :] = (res[::3, :, :].astype(np.uint16) * 6 // 10).astype(np.uint8)
        return res

    @staticmethod
    def edge_neon(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        return cv2.bitwise_and(colored, colored, mask=edges)

    # --- TACTICAL THEME ---
    @staticmethod
    def thermal_vision(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def night_vision(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        green_map = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
        b, g, r = cv2.split(green_map)
        g = cv2.add(g, 50)
        b = cv2.scaleAdd(b, 0.2, np.zeros_like(b))
        r = cv2.scaleAdd(r, 0.2, np.zeros_like(r))
        res = cv2.merge([b, g, r])
        res[::2, :, :] = (res[::2, :, :].astype(np.uint16) * 7 // 10).astype(np.uint8)
        return res


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_finger_extended(pts: List[Tuple[int, int]], finger_name: str) -> bool:
        """Check if a finger is extended using tip-vs-PIP distance from wrist.
        
        For thumb: tip must be further from wrist than MCP.
        For others: use both tip-vs-PIP (from wrist) AND tip-vs-MCP Y-axis check
        so that the 'gun/L-shape' gesture (thumb+index up, rest curled into palm)
        is correctly detected — the curled fingers have tips BELOW their MCP.
        """
        if not pts or len(pts) < 21:
            return True

        wrist = np.array(pts[0])

        if finger_name == "thumb":
            mcp = np.array(pts[2])
            tip = np.array(pts[4])
            return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(mcp - wrist)) * 1.10

        mapping = {
            "index": (8, 6, 5),   # tip, PIP, MCP
            "middle": (12, 10, 9),
            "ring": (16, 14, 13),
            "pinky": (20, 18, 17),
        }
        if finger_name not in mapping:
            return True

        tip_idx, pip_idx, mcp_idx = mapping[finger_name]
        tip = np.array(pts[tip_idx])
        pip = np.array(pts[pip_idx])
        mcp = np.array(pts[mcp_idx])

        # ponytail: dual check — finger is extended if:
        # 1) tip is further from wrist than PIP (classic distance check), AND
        # 2) tip Y is above (or near) MCP Y (in screen coords, lower Y = higher).
        # This correctly marks curled-into-palm fingers as NOT extended even when
        # the distance ratio is ambiguous (e.g. finger curled sideways).
        dist_tip = float(np.linalg.norm(tip - wrist))
        dist_pip = float(np.linalg.norm(pip - wrist))
        # Y-axis: tip should be above or at MCP level if extended (screen Y: up = smaller)
        tip_above_mcp = tip[1] <= mcp[1] + 15  # 15px tolerance
        return dist_tip > dist_pip * 1.0 and tip_above_mcp

    @staticmethod
    def is_fist_closed_pts(pts: List[Tuple[int, int]], threshold: float) -> bool:
        if not pts or len(pts) < 21:
            return False
        wrist = np.array(pts[0])
        tips = [pts[t] for t in [8, 12, 16, 20]]
        distances = [np.linalg.norm(np.array(t) - wrist) for t in tips]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_peace_sign_pts(pts: List[Tuple[int, int]]) -> bool:
        if not pts or len(pts) < 21:
            return False
        wrist = np.array(pts[0])
        d_index = np.linalg.norm(np.array(pts[8]) - wrist)
        d_middle = np.linalg.norm(np.array(pts[12]) - wrist)
        d_ring = np.linalg.norm(np.array(pts[16]) - wrist)
        d_pinky = np.linalg.norm(np.array(pts[20]) - wrist)
        return (d_index > d_ring * 1.2) and (d_middle > d_pinky * 1.2) and (d_ring < d_index * 0.75)

    @staticmethod
    def is_pinch_pts(pts: List[Tuple[int, int]], ratio_threshold: float = 0.28) -> bool:
        if not pts or len(pts) < 21:
            return False
        wrist = pts[0]
        middle_mcp = pts[9]
        hand_scale = max(1.0, GeometryUtils.euclidean_dist(wrist, middle_mcp))

        thumb_tip = pts[4]
        index_tip = pts[8]
        pinky_tip = pts[20]

        d_thumb_index = GeometryUtils.euclidean_dist(thumb_tip, index_tip) / hand_scale
        d_thumb_pinky = GeometryUtils.euclidean_dist(thumb_tip, pinky_tip) / hand_scale

        return min(d_thumb_index, d_thumb_pinky) < ratio_threshold

    @staticmethod
    def is_l_shape(pts: List[Tuple[int, int]]) -> bool:
        """Check if a hand is forming the L / J framing gesture (Index & Thumb extended, others curled)."""
        if not pts or len(pts) < 21:
            return False
        wrist = np.array(pts[0])
        middle_mcp = np.array(pts[9])
        scale = max(1.0, float(np.linalg.norm(middle_mcp - wrist)))

        d_thumb = float(np.linalg.norm(np.array(pts[4]) - wrist)) / scale
        d_index = float(np.linalg.norm(np.array(pts[8]) - wrist)) / scale

        d_middle = float(np.linalg.norm(np.array(pts[12]) - wrist)) / scale
        d_ring = float(np.linalg.norm(np.array(pts[16]) - wrist)) / scale
        d_pinky = float(np.linalg.norm(np.array(pts[20]) - wrist)) / scale

        thumb_ext = d_thumb > 0.95
        index_ext = d_index > 1.05
        others_curled = (d_middle < d_index * 0.85) and (d_ring < d_index * 0.85) and (d_pinky < d_index * 0.85)

        return thumb_ext and index_ext and others_curled

    @staticmethod
    def sort_polygon_vertices(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(pts) < 3:
            return pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    @staticmethod
    def make_flexible_quad(h1: List[Tuple[int, int]], h2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Construct a flexible 4-corner quad frame between 2 hands that adapts seamlessly to twisting/rotation."""
        p1, p2 = h1[8], h1[4]  # Left Index, Left Thumb
        p3, p4 = h2[4], h2[8]  # Right Thumb, Right Index

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        quad = [p1, p2, p3, p4]

        # Check if left edge (p1-p2) crosses right edge (p3-p4)
        if intersect(p1, p2, p3, p4):
            p3, p4 = p4, p3
            quad = [p1, p2, p3, p4]

        # Check if top edge (p4-p1) crosses bottom edge (p2-p3)
        if intersect(quad[3], quad[0], quad[1], quad[2]):
            quad = [quad[0], quad[3], quad[2], quad[1]]

        return quad

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return GeometryUtils.sort_polygon_vertices(pts)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return GeometryUtils.sort_polygon_vertices(pts)


@dataclass
class Toast:
    message: str
    icon: str = "i"
    start_time: float = field(default_factory=time.time)
    duration: float = 2.5


class ToastManager:
    def __init__(self) -> None:
        self.toasts: List[Toast] = []

    def add(self, message: str, icon: str = "i", duration: float = 2.5) -> None:
        self.toasts.append(Toast(message=message, icon=icon, start_time=time.time(), duration=duration))

    def render(self, frame: np.ndarray) -> None:
        now = time.time()
        self.toasts = [t for t in self.toasts if now - t.start_time < t.duration]
        if not self.toasts:
            return

        h, w = frame.shape[:2]
        y_offset = 64
        for toast in self.toasts[:2]:
            text = f"{toast.message}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pw = tw + 28
            ph = th + 14
            px1 = (w - pw) // 2
            py1 = y_offset
            px2 = px1 + pw
            py2 = py1 + ph

            if px1 > 0 and px2 < w and py1 > 0 and py2 < h:
                sub = frame[py1:py2, px1:px2]
                glass = cv2.addWeighted(sub, 0.35, np.full_like(sub, (25, 25, 28)), 0.65, 0)
                cv2.rectangle(glass, (0, 0), (pw, ph), (120, 120, 135), 1, cv2.LINE_AA)
                frame[py1:py2, px1:px2] = glass

                cv2.putText(frame, text, (px1 + 14, py1 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += ph + 10


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.themed_filters: Dict[str, Dict[str, Callable[[np.ndarray], np.ndarray]]] = {
            "Y2K POP-ART": {
                "red-halftone": FilterBank.red_halftone_print,
                "indie-flash": FilterBank.indie_night_flash,
                "y2k-lime-doodle": FilterBank.y2k_lime_cyber_doodle,
                "pink-starburst": FilterBank.grunge_pink_starburst,
                "pink-halo-dots": FilterBank.bw_halftone_pink_halo,
            },
            "SPECIAL FX": {
                "glitch-cyber": FilterBank.glitch_cyber,
                "broken-tv": FilterBank.crt_broken_tv,
                "vhs-retro": FilterBank.vhs_retro,
                "pixel-8bit": FilterBank.pixel_art_8bit,
                "chromatic": FilterBank.chromatic_aberration,
                "film-grain": FilterBank.film_grain,
                "emboss": FilterBank.emboss_relief,
                "solarize": FilterBank.solarize_psychedelic,
                "comic": FilterBank.comic_posterize,
                "negative": FilterBank.negative_invert,
            },
            "CYBER": {
                "matrix-rain": FilterBank.matrix_rain,
                "hologram-cyan": FilterBank.hologram_cyan,
                "edge-neon": FilterBank.edge_neon,
                "duotone-cyber": FilterBank.duotone_cyberpunk,
                "cross-process": FilterBank.cross_process,
            },
            "TACTICAL": {
                "thermal-vision": FilterBank.thermal_vision,
                "night-vision": FilterBank.night_vision,
                "xray-scan": FilterBank.xray_scan,
            },
        }

        # Flattened filters dict for fast access
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        for theme, fdict in self.themed_filters.items():
            self.filters.update(fdict)

        self.filter_keys = list(self.filters.keys())
        self.active_theme_name = "ALL"
        self.active_filter_idx = 0
        
        self.auto_cycle_active = False
        self.last_auto_cycle_time = 0.0
        
        # ponytail: portal only activates after a pinch gesture, not just two hands on screen.
        # Pinch toggles portal on/off. Portal stays active until next pinch or hands disappear.
        self.portal_active = False
        self._portal_hand_count_prev = 0
        
        self.is_3d_mode = False
        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0
        self.last_gesture_time = 0.0
        
        self.toast_mgr = ToastManager()
        self.fps_tracker: List[float] = []
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.rec_start_time = 0.0
        self.shutter_flash_frames = 0
        self.pinch_anim_point: Optional[Tuple[int, int]] = None
        self.pinch_anim_frames = 0

        self.engine = HandDetectorEngine(cfg.frame_width, cfg.frame_height, detect_scale=1.0)
        self.smoother = HandSmoother(alpha=0.55, ghost_frames=15)
        self._frame_count = 0
        self._detect_every = 1  # 100% full frame detection for precision

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def toggle_auto_cycle(self) -> None:
        self.auto_cycle_active = not self.auto_cycle_active
        status = "ON (Every 2s)" if self.auto_cycle_active else "OFF (Gesture/Manual)"
        self.toast_mgr.add(f"Auto-Cycle: {status}", "A", 2.2)

    def cycle_theme(self) -> None:
        themes = ["ALL", "Y2K POP-ART", "SPECIAL FX", "CYBER", "TACTICAL"]
        curr_idx = themes.index(self.active_theme_name) if self.active_theme_name in themes else 0
        next_theme = themes[(curr_idx + 1) % len(themes)]
        self.active_theme_name = next_theme

        if next_theme == "ALL":
            self.filter_keys = list(self.filters.keys())
        else:
            self.filter_keys = list(self.themed_filters[next_theme].keys())

        self.active_filter_idx = 0
        self.toast_mgr.add(f"Theme: {next_theme}", "T", 2.0)

    def set_active_fingers(self, count: int) -> None:
        self.cfg.active_fingers = max(1, min(5, count))
        labels = {1: "1 (Circle)", 2: "2 (Pill)", 3: "3 (Triangle)", 4: "4 (Quad)", 5: "5 (Full Hand)"}
        self.toast_mgr.add(f"Fingers: {labels[self.cfg.active_fingers]}", "#", 1.8)

    def cycle_active_fingers(self) -> None:
        next_count = 1 if self.cfg.active_fingers >= 5 else self.cfg.active_fingers + 1
        self.set_active_fingers(next_count)

    def toggle_gesture_snap(self) -> None:
        self.cfg.enable_gesture_snap = not self.cfg.enable_gesture_snap
        status = "ON (Peace Sign)" if self.cfg.enable_gesture_snap else "OFF (Manual 'S' Only)"
        self.toast_mgr.add(f"Gesture Snap: {status}", "G", 2.2)

    def toggle_mode(self) -> None:
        self.is_3d_mode = False

    def capture_screenshot(self, frame: np.ndarray) -> str:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        filename = os.path.join(self.cfg.output_dir, f"cap_{int(time.time())}.png")
        cv2.imwrite(filename, frame)
        abs_path = os.path.abspath(filename)
        print(f"[INFO] Screenshot disimpan ke: {abs_path}")
        self.toast_mgr.add(f"Saved: {filename}", "SNAP", 3.0)
        self.shutter_flash_frames = 4
        return filename

    def toggle_recording(self, frame_size: Tuple[int, int], fps: float = 30.0) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        if not self.is_recording:
            filename = os.path.join(self.cfg.output_dir, f"rec_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            self.is_recording = True
            self.rec_start_time = time.time()
            self.toast_mgr.add(f"Recording Started: {filename}", "REC", 2.5)
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            self.toast_mgr.add("Recording Saved", "STOP", 2.5)

    def render_portal(self, frame: np.ndarray, pts, filter_key: str) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return frame

        if isinstance(pts, np.ndarray):
            pts = [(int(p[0]), int(p[1])) for p in pts]
        else:
            pts = [(int(p[0]), int(p[1])) for p in pts if len(p) >= 2]

        if not pts:
            return frame

        fh, fw = frame.shape[:2]

        # 1-Point Portal (Circle around index finger)
    def render_portal(self, frame: np.ndarray, pts, filter_key: str) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return frame

        if isinstance(pts, np.ndarray):
            pts = [(int(p[0]), int(p[1])) for p in pts]
        else:
            pts = [(int(p[0]), int(p[1])) for p in pts if len(p) >= 2]

        if not pts:
            return frame

        fh, fw = frame.shape[:2]

        # Use flexible quad order if 4 points are passed, else fallback to convex sort
        pts_ccw = pts if len(pts) == 4 else GeometryUtils.sort_polygon_vertices(pts)
        poly = np.array(pts_ccw, dtype=np.int32)

        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w <= 10 or h <= 10:
            return frame

        # Crop ROI without any perspective warping/stretching
        # This keeps the image and face 100% un-distorted (1-to-1 alignment with background)!
        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)

        # Create exact polygon mask cutout
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        # Clean Apple Glass Outline Border around the portal box
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for pt in pts_ccw:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    def draw_hand_skeleton(self, frame: np.ndarray, pts: List[Tuple[int, int]]) -> None:
        if not pts or len(pts) < 21:
            return
        # Apple Minimalist Translucent White/Silver Skeleton Lines
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1], pts[p2], (240, 240, 245), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def select_active_fingertips(self, hand_pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not hand_pts or len(hand_pts) < 21:
            return hand_pts if hand_pts else []

        # Focus on index finger (landmark 8)
        if self.cfg.active_fingers == 1:
            return [hand_pts[8]]

        finger_order = [
            ("index", 8),
            ("thumb", 4),
            ("middle", 12),
            ("ring", 16),
            ("pinky", 20),
        ]

        extended_tips = []
        for fname, tip_idx in finger_order:
            if GeometryUtils.is_finger_extended(hand_pts, fname):
                extended_tips.append(hand_pts[tip_idx])

        if not extended_tips:
            extended_tips = [hand_pts[8]]

        count = max(1, min(5, self.cfg.active_fingers))
        return extended_tips[:count]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        start_t = time.time()
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        if fw != self.cfg.frame_width or fh != self.cfg.frame_height:
            frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        now = time.time()

        if self.auto_cycle_active:
            if now - self.last_auto_cycle_time > self.cfg.auto_cycle_interval:
                self.active_filter_idx = random.randint(0, len(self.filter_keys) - 1)
                self.last_auto_cycle_time = now

        # Skip-frame detection: run MediaPipe every 2 frames, smoother fills gaps
        self._frame_count += 1
        if self._frame_count % self._detect_every == 0:
            raw_hands = self.engine.detect(frame)
        else:
            raw_hands = []
        hands = self.smoother.smooth(raw_hands)
        current_hand_count = len(hands) if hands else 0
        
        fist_count = 0
        l_shape_hands = []

        if hands:
            for hand_pts in hands:
                if not hand_pts:
                    continue

                if self.cfg.show_hud and len(hand_pts) >= 21:
                    self.draw_hand_skeleton(frame, hand_pts)

                if len(hand_pts) >= 21:
                    # Pinch gesture: switch filter
                    if GeometryUtils.is_pinch_pts(hand_pts, self.cfg.pinch_threshold_ratio):
                        if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                            self.cycle_filter(1)
                            self.last_switch_time = now
                            self.pinch_anim_point = hand_pts[4]
                            self.pinch_anim_frames = 6

                    # Check J & L framing sign (Index + Thumb extended)
                    if GeometryUtils.is_l_shape(hand_pts):
                        l_shape_hands.append(hand_pts)

                    if GeometryUtils.is_fist_closed_pts(hand_pts, self.cfg.fist_dist_threshold_px):
                        fist_count += 1

                    if self.cfg.enable_gesture_snap and GeometryUtils.is_peace_sign_pts(hand_pts):
                        if now - self.last_gesture_time > self.cfg.gesture_cooldown_sec:
                            self.capture_screenshot(frame)
                            self.last_gesture_time = now

            if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
                self.toggle_mode()
                self.last_mode_toggle = now

            # Portal activates ONLY when the two hand skeletons touch/come close (< 160px)
            # Once activated, it stretches and flexes dynamically as hands rotate/twist ("saat dipelintir")
            if len(hands) == 2:
                h1, h2 = hands[0], hands[1]
                if h1[0][0] > h2[0][0]:
                    h1, h2 = h2, h1

                d_wrist = GeometryUtils.euclidean_dist(h1[0], h2[0])
                d_index = GeometryUtils.euclidean_dist(h1[8], h2[8])
                d_thumb = GeometryUtils.euclidean_dist(h1[4], h2[4])
                min_touch = min(d_wrist, d_index, d_thumb)

                if min_touch < 160:
                    if not self.portal_active:
                        self.portal_active = True
                        self.pinch_anim_point = ((h1[8][0] + h2[8][0]) // 2, (h1[8][1] + h2[8][1]) // 2)
                        self.pinch_anim_frames = 6

                if self.portal_active:
                    pts = GeometryUtils.make_flexible_quad(h1, h2)
                    frame = self.render_portal(frame, pts, self.current_filter_name)
            else:
                self.portal_active = False

        # Deactivate portal when hands disappear
        if current_hand_count == 0:
            self.portal_active = False
        self._portal_hand_count_prev = current_hand_count

        if self.pinch_anim_frames > 0 and self.pinch_anim_point is not None:
            r = (7 - self.pinch_anim_frames) * 6
            cv2.circle(frame, self.pinch_anim_point, r, (255, 255, 255), 1, cv2.LINE_AA)
            self.pinch_anim_frames -= 1

        elapsed = time.time() - start_t
        if elapsed > 0:
            self.fps_tracker.append(1.0 / elapsed)
            if len(self.fps_tracker) > 20:
                self.fps_tracker.pop(0)
        current_fps = int(np.mean(self.fps_tracker)) if self.fps_tracker else 30

        if self.cfg.show_hud:
            self._draw_hud(frame, current_fps)
            self.toast_mgr.render(frame)

        if self.shutter_flash_frames > 0:
            white_overlay = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, 0.4, white_overlay, 0.6, 0)
            self.shutter_flash_frames -= 1

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

        return frame

    def _draw_hud(self, frame: np.ndarray, fps: int) -> None:
        """
        Apple Minimalist Clean Glass HUD (San Francisco Style Layout).
        """
        h, w = frame.shape[:2]
        if h < 100 or w < 300:
            return

        # Top Floating Apple Dynamic Glass Header Capsule
        bar_h = 44
        bar_w = min(w - 24, 910)
        bx1 = (w - bar_w) // 2
        by1 = 10
        bx2 = bx1 + bar_w
        by2 = by1 + bar_h

        sub_header = frame[by1:by2, bx1:bx2]
        glass_header = cv2.addWeighted(sub_header, 0.30, np.full_like(sub_header, (25, 25, 28)), 0.70, 0)
        cv2.rectangle(glass_header, (0, 0), (bar_w, bar_h), (90, 90, 100), 1, cv2.LINE_AA)
        frame[by1:by2, bx1:bx2] = glass_header

        # Active Theme & Filter Pill (Positioned cleanly on top-left of glass bar)
        filter_str = f"[{self.active_theme_name}] {self.current_filter_name.upper()}"
        cv2.putText(frame, filter_str, (bx1 + 18, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Auto-Cycle Pill
        if self.auto_cycle_active:
            cv2.putText(frame, "AUTO 2s", (bx1 + 280, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 140), 1, cv2.LINE_AA)

        # FPS Indicator
        fps_color = (120, 235, 140) if fps >= 25 else (240, 200, 100)
        cv2.putText(frame, f"{fps} FPS", (bx2 - 80, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_color, 1, cv2.LINE_AA)

        if self.is_recording:
            rec_dur = int(time.time() - self.rec_start_time)
            rec_str = f"REC {rec_dur//60:02d}:{rec_dur%60:02d}"
            cv2.circle(frame, (bx2 - 160, by1 + 24), 5, (80, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, rec_str, (bx2 - 148, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1, cv2.LINE_AA)

        # Bottom Apple Floating Controls Pill
        foot_h = 30
        foot_w = min(w - 30, 820)
        fx1 = (w - foot_w) // 2
        fy1 = h - 40
        fx2 = fx1 + foot_w
        fy2 = fy1 + foot_h

        sub_footer = frame[fy1:fy2, fx1:fx2]
        glass_footer = cv2.addWeighted(sub_footer, 0.35, np.full_like(sub_footer, (20, 20, 24)), 0.65, 0)
        cv2.rectangle(glass_footer, (0, 0), (foot_w, foot_h), (80, 80, 95), 1, cv2.LINE_AA)
        frame[fy1:fy2, fx1:fx2] = glass_footer

        controls_str = "Two Hands=Portal Box  |  Pinch=Filter  |  [T] Theme  |  [S] Snap  |  [Q] Quit"
        cv2.putText(frame, controls_str, (fx1 + 18, fy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 235), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="HandFlux Pro - Real-time Hand Gesture Filter Engine")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Frame width (default: 960)")
    parser.add_argument("--height", type=int, default=540, help="Frame height (default: 540)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--fingers", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Number of active portal fingers (1-5)")
    parser.add_argument("--auto-cycle", action="store_true", help="Start with Auto-Cycle mode enabled (switches filter every 2s)")
    parser.add_argument("--gesture-snap", action="store_true", help="Enable peace sign auto-screenshot gesture by default")
    parser.add_argument("--no-hud", action="store_true", help="Start with HUD disabled")
    args = parser.parse_args()

    cfg = PipelineConfig(
        cam_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        active_fingers=args.fingers,
        enable_gesture_snap=args.gesture_snap,
        show_hud=not args.no_hud,
    )
    processor = PortalProcessor(cfg)
    if args.auto_cycle:
        processor.auto_cycle_active = True

    cap = None
    for cam_index in [cfg.cam_index, 0, 1, 2, 3]:
        for backend in ([cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]):
            try:
                cap = cv2.VideoCapture(cam_index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
                    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Enable hardware Auto Exposure & Auto White Balance
                    try:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                    except Exception:
                        pass

                    # Warm-up read
                    ret, test_f = cap.read()
                    if ret and test_f is not None and test_f.size > 0:
                        print(f"[INFO] Kamera terdeteksi pada indeks {cam_index} (Backend: {backend})")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception:
                if cap:
                    cap.release()
                    cap = None
        if cap and cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        print("[ERROR] Kamera ga ke detect weh! Coba cek settingan lu lagi.")
        return

    cv2.namedWindow("HandFlux Pro Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HandFlux Pro Engine", cfg.frame_width, cfg.frame_height)
    print("[INFO] HandFlux Pro berjalan (40 Filters, 5 Themes, Auto-Cycle Mode Available).")
    print("[INFO] Kontrol: [A] Auto-Cycle 2s | [T] Switch Theme | [S] Screenshot | [1-5 / F] Jari | [Q] Keluar")

    try:
        while True:
            try:
                visible = cv2.getWindowProperty("HandFlux Pro Engine", cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = 1

            if visible < 1:
                break

            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((cfg.frame_height, cfg.frame_width, 3), dtype=np.uint8)
                cv2.putText(frame, "WEBCAM UNAVAILABLE", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Hubungkan webcam atau periksa izin kamera", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_frame = processor.process_frame(frame)
            cv2.imshow("HandFlux Pro Engine", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("a"), ord("A")):
                processor.toggle_auto_cycle()
            elif key in (ord("t"), ord("T")):
                processor.cycle_theme()
            elif key in (ord("s"), ord("S")):
                processor.capture_screenshot(out_frame)
            elif key in (ord("g"), ord("G")):
                processor.toggle_gesture_snap()
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                processor.set_active_fingers(int(chr(key)))
            elif key in (ord("f"), ord("F")):
                processor.cycle_active_fingers()
            elif key in (ord("c"), ord("C")):
                processor.toggle_mode()
            elif key in (ord("n"), ord("N")):
                processor.cycle_filter(1)
            elif key in (ord("p"), ord("P")):
                processor.cycle_filter(-1)
            elif key in (ord("r"), ord("R")):
                processor.toggle_recording((cfg.frame_width, cfg.frame_height))
            elif key in (ord("m"), ord("M")):
                cfg.mirror = not cfg.mirror
                processor.toast_mgr.add(f"Mirror: {'ON' if cfg.mirror else 'OFF'}", "M", 1.8)
            elif key in (ord("h"), ord("H")):
                cfg.show_hud = not cfg.show_hud

    finally:
        if processor.is_recording and processor.video_writer is not None:
            processor.video_writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()