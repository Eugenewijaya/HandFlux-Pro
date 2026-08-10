"""
HandFlux Pro - Real-time Hand Gesture Filter & Portal Engine
"""

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
        """Reorder new_hands to best match prev_hands by nearest centroid."""
        if len(self.prev_hands) < 2 or len(new_hands) < 2:
            return new_hands
        prev_c = [self._centroid(h) for h in self.prev_hands]
        new_c = [self._centroid(h) for h in new_hands]
        # Try both orderings, pick the one with smallest total distance
        d_same = sum(np.hypot(prev_c[i][0] - new_c[i][0], prev_c[i][1] - new_c[i][1]) for i in range(2))
        d_swap = sum(np.hypot(prev_c[i][0] - new_c[1-i][0], prev_c[i][1] - new_c[1-i][1]) for i in range(2))
        if d_swap < d_same:
            return [new_hands[1], new_hands[0]]
        return new_hands

    def smooth(self, hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if not hands:
            # Ghost frames: extrapolate from velocity instead of freezing
            if self.prev_hands and self._miss_count < self.ghost_frames:
                self._miss_count += 1
                # ponytail: decay velocity each ghost frame to avoid runaway drift
                decay = max(0.0, 1.0 - self._miss_count * 0.12)
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
                    sx = self.alpha * curr[0] + (1 - self.alpha) * prev[j][0]
                    sy = self.alpha * curr[1] + (1 - self.alpha) * prev[j][1]
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
        # ponytail: detect at half resolution for ~2x speed. Landmarks are
        # 0-1 normalized so they map correctly to full-res display coords.
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

                # ponytail: aggressive detection thresholds — we'd rather get
                # occasional false positives than lose a real hand mid-gesture.
                # Tracking confidence low too, so MediaPipe keeps tracking
                # instead of falling back to the slower detection path.
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.35,
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
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.35,
                    )
                    self.mode = "solutions"
                    print("[INFO] Engine Hand Tracking: MediaPipe Solutions (Legacy)")
            except Exception as e:
                print(f"[DEBUG] MediaPipe Solutions init failed: {e}")

        if self.mode == "none":
            print("[WARNING] Engine Hand Tracking: Fallback Skin Detector")

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
                        hands_list.append(pts)
            except Exception as e:
                print(f"[ERROR] Tasks detection error: {e}")

        elif self.mode == "solutions" and self.solutions_detector is not None:
            try:
                res = self.solutions_detector.process(rgb)
                if res and res.multi_hand_landmarks:
                    for hand_lm in res.multi_hand_landmarks:
                        pts = [(int(lm.landmark[i].x * self.w), int(lm.landmark[i].y * self.h)) for i in range(21)]
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
    active_fingers: int = 5
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


class FilterBank:
    """Collection of 32 real-time video filters categorized into Themes."""

    # --- CINEMATIC THEME (8 Filters) ---
    @staticmethod
    def teal_orange(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        b = np.clip(b * 1.25 + 15, 0, 255)
        r = np.clip(r * 1.3 + 20, 0, 255)
        g = np.clip(g * 0.85, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def kodachrome(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        r = np.clip(r * 1.2, 0, 255)
        g = np.clip(g * 1.05 + 10, 0, 255)
        b = np.clip(b * 0.9, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def technicolor(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        r = np.clip(r * 1.4, 0, 255)
        g = np.clip(g * 1.2, 0, 255)
        b = np.clip(b * 0.8, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def noir_film(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cinematic_warm(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        r = np.clip(r * 1.15 + 10, 0, 255)
        g = np.clip(g * 1.05 + 5, 0, 255)
        b = np.clip(b * 0.85, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def vignette_cinema(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        kernel_x = cv2.getGaussianKernel(w, w / 2.5)
        kernel_y = cv2.getGaussianKernel(h, h / 2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / np.max(kernel)
        mask_3c = cv2.merge([mask, mask, mask])
        return np.clip(roi * mask_3c, 0, 255).astype(np.uint8)

    @staticmethod
    def sepia(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        res = cv2.transform(roi, kernel)
        return np.clip(res, 0, 255).astype(np.uint8)

    @staticmethod
    def detail_enhance(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return roi
        try:
            return cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
        except Exception:
            return roi

    # --- ANIME & CARTOON THEME (8 Filters) ---
    @staticmethod
    def anime_cel(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        bila = cv2.bilateralFilter(roi, 9, 75, 75)
        gray = cv2.cvtColor(bila, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = np.clip(bila // 32 * 32, 0, 255)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def manga_ink(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cartoon_classic(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 7, 200, 200)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def pop_art(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        h_half, w_half = max(1, h // 2), max(1, w // 2)
        small = cv2.resize(roi, (w_half, h_half))
        t1 = FilterBank.dual_tone(small)
        t2 = cv2.applyColorMap(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_OCEAN)
        t3 = cv2.applyColorMap(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_PINK)
        t4 = FilterBank.thermal(small)
        top = np.hstack([t1, t2])
        bottom = np.hstack([t3, t4])
        return cv2.resize(np.vstack([top, bottom]), (w, h))

    @staticmethod
    def pencil_sketch(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch_img = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pencil_color(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return roi
        try:
            _, color_sketch = cv2.pencilSketch(roi, sigma_s=15, sigma_r=0.07, shade_factor=0.04)
            return color_sketch
        except Exception:
            return roi

    @staticmethod
    def stylized_water(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return roi
        try:
            return cv2.stylization(roi, sigma_s=15, sigma_r=0.3)
        except Exception:
            return roi

    @staticmethod
    def posterize(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        n_colors = 4
        return np.clip((roi // (256 // n_colors)) * (256 // n_colors), 0, 255).astype(np.uint8)

    # --- CYBER & SCI-FI THEME (8 Filters) ---
    @staticmethod
    def cyberpunk(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        r = np.clip(r * 1.3 + 20, 0, 255)
        b = np.clip(b * 1.4 + 30, 0, 255)
        g = np.clip(g * 0.7, 0, 255)
        res = cv2.merge([b, g, r]).astype(np.uint8)
        return cv2.addWeighted(res, 0.85, FilterBank.edge_neon(roi), 0.15, 0)

    @staticmethod
    def matrix(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        green = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
        b, g, r = cv2.split(green)
        g = cv2.add(g, 40)
        return cv2.merge([b, g, r])

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def night_vision(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        green = cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)
        b, g, r = cv2.split(green)
        g = np.clip(g * 1.3, 0, 255).astype(np.uint8)
        return cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])

    @staticmethod
    def hologram(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        b = np.clip(b * 1.5 + 40, 0, 255)
        g = np.clip(g * 1.1 + 20, 0, 255)
        r = np.clip(r * 0.3, 0, 255)
        out = cv2.merge([b, g, r]).astype(np.uint8)
        scanlines = np.ones_like(out)
        scanlines[::3, :, :] = 160
        return cv2.multiply(out, scanlines // 255)

    @staticmethod
    def glitch_rgb(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def anaglyph_3d(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        r_shifted = np.roll(r, 8, axis=1)
        b_shifted = np.roll(b, -8, axis=1)
        return cv2.merge([b_shifted, g, r_shifted])

    @staticmethod
    def emboss_3d(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        kernel = np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ])
        embossed = cv2.filter2D(roi, -1, kernel) + 128
        return np.clip(embossed, 0, 255).astype(np.uint8)

    # --- ARTISTIC & EFX THEME (8 Filters) ---
    @staticmethod
    def oil_paint(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        small = cv2.resize(roi, (max(1, roi.shape[1] // 2), max(1, roi.shape[0] // 2)))
        blur = cv2.medianBlur(small, 7)
        return cv2.resize(blur, (roi.shape[1], roi.shape[0]))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)

    @staticmethod
    def edge_neon(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def vhs_tape(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        r = np.roll(r, 3, axis=1)
        b = np.roll(b, -3, axis=0)
        merged = cv2.merge([b, g, r])
        scanlines = np.ones_like(merged)
        scanlines[::4, :, :] = 180
        out = cv2.multiply(merged, scanlines // 255)
        cv2.putText(out, "PLAY  >>", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out

    @staticmethod
    def solarize(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        threshold = 128
        out = roi.copy()
        out[roi > threshold] = 255 - out[roi > threshold]
        return out

    @staticmethod
    def duotone_cyan(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (255, 240, 0)
        out[mask == 0] = (150, 0, 180)
        return out

    @staticmethod
    def cross_process(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi.astype(np.float32))
        r = np.clip(r * 1.3, 0, 255)
        g = np.clip(g * 1.1, 0, 255)
        b = np.clip(b * 0.7 + 20, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi if roi is not None and roi.size > 0 else roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        return cv2.GaussianBlur(roi, (25, 25), 0)

    # --- EXOTIC THEME (8 Filters) ---
    # These go beyond color grading — spatial distortion, generative art,
    # hardware simulation. All vectorized for real-time ROI performance.

    @staticmethod
    def pixel_sort(roi: np.ndarray) -> np.ndarray:
        """Glitch art: sort pixel rows by brightness in bands."""
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        out = roi.copy()
        band = max(1, h // 12)
        for y0 in range(0, h, band * 2):
            y1 = min(y0 + band, h)
            strip = out[y0:y1].reshape(-1, 3)
            lum = gray[y0:y1].ravel()
            order = np.argsort(lum)
            out[y0:y1] = strip[order].reshape(y1 - y0, w, 3)
        return out

    @staticmethod
    def kaleidoscope(roi: np.ndarray) -> np.ndarray:
        """4-way mirror kaleidoscope — mesmerizing symmetry."""
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi
        qh, qw = h // 2, w // 2
        quad = roi[:qh, :qw]
        top = np.hstack([quad, cv2.flip(quad, 1)])
        full = np.vstack([top, cv2.flip(top, 0)])
        return cv2.resize(full, (w, h))

    @staticmethod
    def water_ripple(roi: np.ndarray) -> np.ndarray:
        """Animated sine-wave displacement — like looking through water."""
        h, w = roi.shape[:2]
        if h < 8 or w < 8:
            return roi
        t = time.time() * 3.0
        y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = x_idx + 6.0 * np.sin(2.0 * np.pi * y_idx / 45.0 + t)
        map_y = y_idx + 6.0 * np.cos(2.0 * np.pi * x_idx / 45.0 + t)
        return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def frosted_glass(roi: np.ndarray) -> np.ndarray:
        """Random neighbor displacement — frosted window effect."""
        h, w = roi.shape[:2]
        if h < 8 or w < 8:
            return roi
        radius = 5
        dy = np.random.randint(-radius, radius + 1, (h, w)).astype(np.float32)
        dx = np.random.randint(-radius, radius + 1, (h, w)).astype(np.float32)
        y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = np.clip(x_idx + dx, 0, w - 1)
        map_y = np.clip(y_idx + dy, 0, h - 1)
        return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    @staticmethod
    def crt_screen(roi: np.ndarray) -> np.ndarray:
        """Full CRT simulation: barrel distortion + scanlines + chromatic aberration."""
        h, w = roi.shape[:2]
        if h < 16 or w < 16:
            return roi
        # Barrel distortion
        cy, cx = h / 2.0, w / 2.0
        y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (x_idx - cx) / cx
        ny = (y_idx - cy) / cy
        r2 = nx * nx + ny * ny
        k = 0.15  # distortion strength
        map_x = (nx * (1 + k * r2) * cx + cx)
        map_y = (ny * (1 + k * r2) * cy + cy)
        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Chromatic aberration
        b, g, r = cv2.split(warped)
        r = np.roll(r, 2, axis=1)
        b = np.roll(b, -2, axis=1)
        out = cv2.merge([b, g, r])
        # Scanlines + phosphor glow
        out[::3, :] = (out[::3, :].astype(np.float32) * 0.65).astype(np.uint8)
        # Slight green tint like old monitors
        out[:, :, 1] = np.clip(out[:, :, 1].astype(np.int16) + 8, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def aurora_gradient(roi: np.ndarray) -> np.ndarray:
        """Map luminance to aurora borealis gradient (teal→green→purple→pink)."""
        if roi is None or roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Build custom LUT: dark=deep purple, mid=teal/green, bright=pink
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.33:
                s = t / 0.33
                lut[i, 0] = [int(80 * s), int(10 * s), int(120 + 40 * s)]  # deep purple
            elif t < 0.66:
                s = (t - 0.33) / 0.33
                lut[i, 0] = [int(80 + 100 * s), int(10 + 200 * s), int(160 - 60 * s)]  # teal/green
            else:
                s = (t - 0.66) / 0.34
                lut[i, 0] = [int(180 - 80 * s), int(210 - 100 * s), int(100 + 155 * s)]  # pink
        return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)

    @staticmethod
    def diamond_mosaic(roi: np.ndarray) -> np.ndarray:
        """Rotated pixelation creating elegant diamond tessellation."""
        h, w = roi.shape[:2]
        if h < 8 or w < 8:
            return roi
        # Rotate 45°, pixelate, rotate back
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        block = 10
        small = cv2.resize(rotated, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        M_inv = cv2.getRotationMatrix2D(center, -45, 1.0)
        return cv2.warpAffine(pixelated, M_inv, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def dream_glow(roi: np.ndarray) -> np.ndarray:
        """Ethereal bloom — bright areas bleed softly into surroundings."""
        if roi is None or roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        if h < 8 or w < 8:
            return roi
        # Extract bright areas, blur them heavily, blend back
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        _, bright_mask = cv2.threshold(v, 160, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(roi, roi, mask=bright_mask)
        k = max(3, (min(h, w) // 4) | 1)  # ensure odd kernel
        glow = cv2.GaussianBlur(bright, (k, k), 0)
        # Warm tint on glow
        glow[:, :, 2] = np.clip(glow[:, :, 2].astype(np.int16) + 30, 0, 255).astype(np.uint8)
        return cv2.addWeighted(roi, 0.7, glow, 0.5, 10)


# ==============================================================================
#  NARUTO NINJUTSU ENGINE (Rasengan 🌀, Fire Style 🔥, Chidori ⚡, Shadow Clone 👥)
# ==============================================================================

class NinjutsuParticle:
    def __init__(self, x: float, y: float, vx: float, vy: float, color: Tuple[int, int, int], life: int, size: int = 3):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = vx, vy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.10
        self.life -= 1

    def draw(self, frame: np.ndarray) -> None:
        if self.life <= 0:
            return
        a = self.life / float(self.max_life)
        sz = max(1, int(self.size * a))
        col = tuple(int(c * a) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), sz, col, -1)


class NinjutsuBlast:
    def __init__(self) -> None:
        self.active = False
        self.cx = 0
        self.cy = 0
        self.start_t = 0.0
        self.duration = 1.6
        self.particles: List[NinjutsuParticle] = []
        self.ring_r = 0

    def activate(self, cx: int, cy: int) -> None:
        self.active = True
        self.cx, self.cy = cx, cy
        self.start_t = time.time()
        self.ring_r = 0
        self.particles = []
        for _ in range(160):
            ang = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 22)
            vx = speed * math.cos(ang)
            vy = speed * math.sin(ang)
            color = random.choice([
                (255, 255, 255), (220, 235, 255),
                (180, 210, 255), (100, 170, 255),
                (60, 140, 255),
            ])
            self.particles.append(
                NinjutsuParticle(cx, cy, vx, vy, color, random.randint(22, 55), random.randint(3, 10))
            )

    def update_and_draw(self, frame: np.ndarray) -> None:
        if not self.active:
            return
        elapsed = time.time() - self.start_t
        if elapsed > self.duration:
            self.active = False
            self.particles = []
            return

        alpha = max(0.0, 1.0 - elapsed / self.duration)
        cx, cy = self.cx, self.cy

        if elapsed < 0.14:
            flash_a = 1.0 - elapsed / 0.14
            ov = frame.copy()
            cv2.circle(ov, (cx, cy), int(320 * (elapsed / 0.14)), (255, 255, 255), -1)
            cv2.addWeighted(ov, flash_a * 0.65, frame, 1 - flash_a * 0.65, 0, frame)

        self.ring_r = int(elapsed / self.duration * 380)
        for offset in [0, 14, 32]:
            r = self.ring_r - offset
            if r > 0:
                ring_a = max(0.0, alpha - offset / 110.0)
                ov = frame.copy()
                cv2.circle(ov, (cx, cy), r, (int(80 * ring_a), int(160 * ring_a), int(255 * ring_a)), max(1, int(5 * ring_a)))
                cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            p.draw(frame)

        if elapsed < 0.65:
            scale = 1.0 + elapsed * 3
            tx = int(cx - 80 * scale * 0.5)
            ty = int(cy - 30)
            cv2.putText(frame, "BLAST!", (tx + 3, ty + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, int(80 * alpha), int(160 * alpha)), 6, cv2.LINE_AA)
            cv2.putText(frame, "BLAST!", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 230, 255), 2, cv2.LINE_AA)


class RasenganEffect:
    def __init__(self, blast_ref: NinjutsuBlast) -> None:
        self.blast = blast_ref
        self.active = False
        self.angle = 0.0
        self.start_time = 0.0
        self.duration = 5.0
        self.cx, self.cy = 0, 0
        self.particles: List[NinjutsuParticle] = []
        self.wind_lines: List[dict] = []
        self.shockwave_r = 0
        self.shockwave_on = False
        self.orb_radius = 0
        self.gyro_rings = [
            (0, 1.0), (math.pi / 3, 1.4), (math.pi * 2 / 3, 0.8),
            (math.pi, 1.2), (math.pi * 4 / 3, 1.6),
        ]

    def activate(self, cx: int, cy: int) -> None:
        self.active = True
        self.cx, self.cy = cx, cy
        self.start_time = time.time()
        self.particles = []
        self.wind_lines = []
        self.orb_radius = 0
        self.shockwave_r = 0
        self.shockwave_on = True
        self.angle = 0.0

    def _glow(self, frame: np.ndarray, cx: int, cy: int, R: int, color: Tuple[int, int, int], layers: int = 14, base_alpha: float = 0.06) -> None:
        for i in range(layers, 0, -1):
            rad = int(R * (1.0 + i * 0.28))
            fade = i / float(layers)
            col = tuple(int(c * fade) for c in color)
            ov = frame.copy()
            cv2.circle(ov, (cx, cy), rad, col, -1)
            cv2.addWeighted(ov, base_alpha * fade, frame, 1 - base_alpha * fade, 0, frame)

    def _draw_gyro_ring(self, frame: np.ndarray, cx: int, cy: int, R: int, phase: float, speed_mul: float, angle_deg: float, alpha: float) -> None:
        arad = math.radians(angle_deg * speed_mul + math.degrees(phase))
        b_ratio = abs(math.sin(arad))
        tilt_rot = math.degrees(arad * 0.5)
        b_axis = max(2, int(R * b_ratio))
        ring_col = (int(120 * alpha), int(190 * alpha), int(255 * alpha))
        cv2.ellipse(frame, (cx, cy), (R, b_axis), int(tilt_rot) % 180, 0, 360, ring_col, 1, cv2.LINE_AA)

    def _spawn_wind_line(self, cx: int, cy: int, R: int) -> dict:
        ang = random.uniform(0, 2 * math.pi)
        dist = random.uniform(R * 2.2, R * 5.0)
        sx = cx + dist * math.cos(ang)
        sy = cy + dist * math.sin(ang)
        end_ang = ang + random.uniform(0.25, 0.6) * random.choice([-1, 1])
        ex = cx + R * 1.05 * math.cos(end_ang)
        ey = cy + R * 1.05 * math.sin(end_ang)
        length_frames = random.randint(6, 14)
        color = random.choice([(160, 200, 255), (200, 220, 255), (100, 160, 255), (255, 255, 255)])
        return {'sx': sx, 'sy': sy, 'ex': ex, 'ey': ey, 'life': length_frames, 'max_life': length_frames, 'color': color}

    def _draw_helical_band(self, frame: np.ndarray, cx: int, cy: int, R: int, arm_index: int, num_arms: int, angle_deg: float, alpha: float, tilt_amp: float = 0.55) -> None:
        pts = []
        base_offset = (360.0 / num_arms) * arm_index
        for t in range(80):
            frac = t / 79.0
            phi = math.radians(angle_deg * 2.2 + base_offset + frac * 340)
            theta = math.radians(frac * 360 * 1.8 + arm_index * 137)
            x3d = R * math.cos(phi) * math.sin(theta)
            y3d = R * tilt_amp * math.sin(phi) * math.sin(theta)
            pts.append((int(cx + x3d), int(cy + y3d)))

        for i in range(len(pts) - 1):
            frac = i / float(len(pts))
            brightness = max(0.0, 1.0 - frac * 0.7)
            line_w = max(1, int(2.5 * brightness * alpha))
            col = (int((80 + 175 * brightness) * alpha), int((150 + 105 * brightness) * alpha), int(255 * alpha))
            cv2.line(frame, pts[i], pts[i + 1], col, line_w, cv2.LINE_AA)

    def update_and_draw(self, frame: np.ndarray, cx: int, cy: int) -> None:
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active = False
            self.particles = []
            self.blast.activate(cx, cy)
            return

        self.cx, self.cy = cx, cy
        alpha = max(0.0, 1.0 - elapsed / self.duration)
        max_r = 64
        self.orb_radius = int(max_r * min(1.0, elapsed / 0.4))
        R = self.orb_radius
        if R < 4:
            return

        spin_speed = 9.0 + 6.0 * min(1.0, elapsed / 0.35)
        self.angle += spin_speed
        h, w = frame.shape[:2]

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (60, 20, 0), -1)
        cv2.addWeighted(ov, 0.07 * alpha, frame, 1 - 0.07 * alpha, 0, frame)

        if self.shockwave_on:
            self.shockwave_r += 22
            ra = max(0.0, 1.0 - self.shockwave_r / 260.0)
            if ra > 0:
                ov = frame.copy()
                cv2.circle(ov, (cx, cy), self.shockwave_r, (int(160 * ra), int(210 * ra), int(255 * ra)), 3)
                cv2.addWeighted(ov, 0.9, frame, 0.1, 0, frame)
            else:
                self.shockwave_on = False

        self._glow(frame, cx, cy, R, (int(60 * alpha), int(130 * alpha), int(255 * alpha)), layers=14, base_alpha=0.06)

        if elapsed < self.duration - 0.4 and random.random() < 0.55:
            self.wind_lines.append(self._spawn_wind_line(cx, cy, R))

        new_wl = []
        for wl in self.wind_lines:
            wl['life'] -= 1
            if wl['life'] > 0:
                fade = (wl['life'] / float(wl['max_life'])) * alpha * 0.7
                col = tuple(int(c * fade) for c in wl['color'])
                cv2.line(frame, (int(wl['sx']), int(wl['sy'])), (int(wl['ex']), int(wl['ey'])), col, 1, cv2.LINE_AA)
                new_wl.append(wl)
        self.wind_lines = new_wl

        pulse = 1.0 + 0.06 * math.sin(elapsed * 16)
        Rp = int(R * pulse)

        ov = frame.copy()
        for layer in range(10, 0, -1):
            r_l = int(Rp * (0.95 + layer * 0.04))
            fade = layer / 10.0
            col = (int(30 * fade), int(80 * fade), int(200 * fade))
            cv2.circle(ov, (cx, cy), r_l, col, -1)
        cv2.circle(ov, (cx, cy), Rp, (50, 130, 240), -1)
        cv2.circle(ov, (cx, cy), int(Rp * 0.78), (90, 170, 255), -1)
        cv2.circle(ov, (cx, cy), int(Rp * 0.56), (160, 210, 255), -1)
        cv2.circle(ov, (cx, cy), int(Rp * 0.38), (220, 238, 255), -1)
        cv2.circle(ov, (cx, cy), int(Rp * 0.22), (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.72 * alpha, frame, 1 - 0.72 * alpha, 0, frame)

        num_arms = 4
        for arm in range(num_arms):
            self._draw_helical_band(frame, cx, cy, Rp, arm, num_arms, self.angle, alpha)

        for phase, speed_mul in self.gyro_rings:
            self._draw_gyro_ring(frame, cx, cy, Rp + 5, phase, speed_mul, self.angle, alpha * 0.75)

        primary_b = int((Rp + 4) * abs(math.sin(math.radians(self.angle * 1.1))))
        primary_b = max(2, primary_b)
        cv2.ellipse(frame, (cx, cy), (Rp + 4, primary_b), int(self.angle * 0.6) % 180, 0, 360, (int(180 * alpha), int(220 * alpha), int(255 * alpha)), 2, cv2.LINE_AA)

        for _ in range(6):
            ang = random.uniform(0, 2 * math.pi)
            tangent_ang = ang + math.pi / 2.0 * random.choice([-1, 1])
            speed = random.uniform(2.5, 6.0)
            sx = cx + int(Rp * math.cos(ang))
            sy = cy + int(Rp * math.sin(ang))
            vx = speed * math.cos(tangent_ang) + random.uniform(-0.8, 0.8)
            vy = speed * math.sin(tangent_ang) + random.uniform(-0.8, 0.8)
            color = random.choice([(255, 255, 255), (220, 235, 255), (160, 200, 255), (100, 160, 255)])
            self.particles.append(NinjutsuParticle(sx, sy, vx, vy, color, random.randint(8, 22), random.randint(1, 3)))

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            p.draw(frame)

        for _ in range(5):
            a1 = random.uniform(0, 2 * math.pi)
            a2 = a1 + random.uniform(-0.7, 0.7)
            ex1, ey1 = int(cx + Rp * math.cos(a1)), int(cy + Rp * math.sin(a1))
            ex2, ey2 = int(cx + Rp * math.cos(a2)), int(cy + Rp * math.sin(a2))
            mx = (ex1 + ex2) // 2 + random.randint(-14, 14)
            my = (ey1 + ey2) // 2 + random.randint(-14, 14)
            col = (int(140 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(frame, (ex1, ey1), (mx, my), col, 1, cv2.LINE_AA)
            cv2.line(frame, (mx, my), (ex2, ey2), col, 1, cv2.LINE_AA)

        time_left = max(0.0, 1.0 - elapsed / self.duration)
        arc_end = int(360 * time_left)
        cv2.ellipse(frame, (cx, cy), (Rp + 14, Rp + 14), -90, 0, arc_end, (int(100 * alpha), int(180 * alpha), int(255 * alpha)), 2, cv2.LINE_AA)

        tx, ty = cx - 72, cy - Rp - 24
        cv2.putText(frame, "RASENGAN!", (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 60, 140), 6, cv2.LINE_AA)
        cv2.putText(frame, "RASENGAN!", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (int(200 * alpha), int(230 * alpha), 255), 2, cv2.LINE_AA)


class ChidoriEffect:
    def __init__(self) -> None:
        self.active = False
        self.cx, self.cy = 0, 0
        self.start_time = 0.0
        self.duration = 4.0

    def activate(self, cx: int, cy: int) -> None:
        self.active = True
        self.cx, self.cy = cx, cy
        self.start_time = time.time()

    def _bolt(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], width: int = 1) -> None:
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy)) // 6
        if steps == 0:
            return
        pts = [(x1, y1)]
        for i in range(1, steps):
            t = i / float(steps)
            pts.append((int(x1 + dx * t + random.randint(-12, 12)), int(y1 + dy * t + random.randint(-12, 12))))
        pts.append((x2, y2))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, width, cv2.LINE_AA)

    def update_and_draw(self, frame: np.ndarray, cx: int, cy: int) -> None:
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active = False
            return
        self.cx, self.cy = cx, cy
        alpha = max(0.0, 1.0 - elapsed / self.duration)
        ov = frame.copy()
        for r in range(50, 10, -8):
            cv2.circle(ov, (cx, cy), r, (255, 255, int(100 * alpha)), -1)
        for ad in range(0, 360, 30):
            a = math.radians(ad + random.randint(-10, 10))
            l = random.randint(40, 80)
            self._bolt(ov, cx, cy, int(cx + l * math.cos(a)), int(cy + l * math.sin(a)), (int(200 * alpha), int(200 * alpha), 255), 1)
        cv2.circle(ov, (cx, cy), 15, (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "CHIDORI!", (cx - 50, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2, cv2.LINE_AA)


class FireballEffect:
    def __init__(self) -> None:
        self.active = False
        self.cx, self.cy = 0, 0
        self.start_time = 0.0
        self.duration = 3.5
        self.particles: List[NinjutsuParticle] = []

    def activate(self, cx: int, cy: int) -> None:
        self.active = True
        self.cx, self.cy = cx, cy
        self.start_time = time.time()
        self.particles = []

    def update_and_draw(self, frame: np.ndarray, cx: int, cy: int) -> None:
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active = False
            self.particles = []
            return
        self.cx, self.cy = cx, cy
        for _ in range(8):
            ang = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append(NinjutsuParticle(
                cx, cy, speed * math.cos(ang), speed * math.sin(ang) - random.uniform(1, 3),
                random.choice([(0, 50, 255), (0, 120, 255), (0, 200, 255), (0, 255, 200)]),
                random.randint(15, 30), random.randint(4, 10)
            ))
        ov = frame.copy()
        for r in [50, 35, 20]:
            cv2.circle(ov, (cx, cy), r, (0, max(0, 255 - r * 3 - 100), 255), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            p.draw(frame)
        cv2.putText(frame, "FIRE STYLE!", (cx - 70, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2, cv2.LINE_AA)


class ShadowCloneEffect:
    def __init__(self) -> None:
        self.active = False
        self.start_time = 0.0
        self.duration = 5.0

    def activate(self) -> None:
        self.active = True
        self.start_time = time.time()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        if time.time() - self.start_time > self.duration:
            self.active = False
            return frame
        h, w = frame.shape[:2]
        clone = cv2.flip(frame.copy(), 1)
        clone[:, :, 0] = np.clip(clone[:, :, 0].astype(int) + 40, 0, 255)
        combined = np.hstack([cv2.resize(frame, (w // 2, h)), cv2.resize(clone, (w // 2, h))])
        cv2.putText(combined, "YOU", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "SHADOW CLONE", (w // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
        cv2.putText(combined, "SHADOW CLONE JUTSU!", (w // 4 - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        return combined


class HandSignClassifier:
    @staticmethod
    def classify_sign(pts: List[Tuple[int, int]]) -> str:
        if not pts or len(pts) < 21:
            return "UNKNOWN"
        def fu(tip_idx: int, pip_idx: int) -> bool:
            return pts[tip_idx][1] < pts[pip_idx][1]

        idx = fu(8, 6)
        mid = fu(12, 10)
        rng = fu(16, 14)
        pnk = fu(20, 18)

        if idx and mid and not rng and not pnk: return "SNAKE"
        if idx and not mid and not rng and pnk: return "RAM"
        if not idx and mid and rng and not pnk: return "BOAR"
        if idx and mid and rng and pnk: return "BIRD"
        if not idx and not mid and not rng and not pnk: return "MONKEY"
        if idx and not mid and not rng and not pnk: return "HORSE"
        if not idx and not mid and rng and pnk: return "DOG"
        if idx and mid and not rng and pnk: return "OX"
        if not idx and mid and not rng and not pnk: return "TIGER"
        if not idx and not mid and not rng and pnk: return "HARE"
        return "UNKNOWN"

    @staticmethod
    def detect_clap(hand_pts_list: List[List[Tuple[int, int]]], clap_dist_px: float = 90.0) -> Optional[Tuple[int, int]]:
        if len(hand_pts_list) < 2:
            return None
        p1 = hand_pts_list[0][9]
        p2 = hand_pts_list[1][9]
        dist = GeometryUtils.euclidean_dist(p1, p2)
        if dist < clap_dist_px:
            cx = (p1[0] + p2[0]) // 2
            cy = (p1[1] + p2[1]) // 2
            return (cx, cy)
        return None


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_finger_extended(pts: List[Tuple[int, int]], finger_name: str) -> bool:
        if not pts or len(pts) < 21:
            return True

        wrist = np.array(pts[0])

        if finger_name == "thumb":
            mcp = np.array(pts[2])
            tip = np.array(pts[4])
            return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(mcp - wrist)) * 1.15

        mapping = {
            "index": (8, 6),
            "middle": (12, 10),
            "ring": (16, 14),
            "pinky": (20, 18),
        }
        if finger_name not in mapping:
            return True

        tip_idx, pip_idx = mapping[finger_name]
        pip = np.array(pts[pip_idx])
        tip = np.array(pts[tip_idx])

        # ponytail: 1.03 (was 1.08) — softer threshold so partially-bent fingers
        # still register as extended.  Folded-flat fingers still fail (tip << pip).
        return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist)) * 1.03

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
    def sort_polygon_vertices(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(pts) < 3:
            return pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

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
        y_offset = 70
        for toast in self.toasts[:3]:
            text = f"[{toast.icon}] {toast.message}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            px1, py1 = max(0, w - tw - 30), max(0, y_offset)
            px2, py2 = min(w, w - 10), min(h, y_offset + th + 16)

            if px2 > px1 + 5 and py2 > py1 + 5:
                sub = frame[py1:py2, px1:px2]
                glass = cv2.addWeighted(sub, 0.3, np.full_like(sub, 20), 0.7, 0)
                cv2.rectangle(glass, (0, 0), (px2 - px1, py2 - py1), (0, 240, 255), 1)
                frame[py1:py2, px1:px2] = glass

                cv2.putText(frame, text, (px1 + 10, py1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += th + 24


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        # 32 Filters mapped by 4 Themes
        self.themed_filters: Dict[str, Dict[str, Callable[[np.ndarray], np.ndarray]]] = {
            "CINEMATIC": {
                "teal-orange": FilterBank.teal_orange,
                "kodachrome": FilterBank.kodachrome,
                "technicolor": FilterBank.technicolor,
                "noir-film": FilterBank.noir_film,
                "cinematic-warm": FilterBank.cinematic_warm,
                "vignette-cinema": FilterBank.vignette_cinema,
                "sepia-vintage": FilterBank.sepia,
                "detail-enhance": FilterBank.detail_enhance,
            },
            "ANIME": {
                "anime-cel": FilterBank.anime_cel,
                "manga-ink": FilterBank.manga_ink,
                "cartoon-classic": FilterBank.cartoon_classic,
                "pop-art": FilterBank.pop_art,
                "pencil-sketch": FilterBank.pencil_sketch,
                "pencil-color": FilterBank.pencil_color,
                "stylized-water": FilterBank.stylized_water,
                "posterize": FilterBank.posterize,
            },
            "CYBER": {
                "cyberpunk": FilterBank.cyberpunk,
                "matrix": FilterBank.matrix,
                "thermal": FilterBank.thermal,
                "night-vision": FilterBank.night_vision,
                "hologram": FilterBank.hologram,
                "glitch-rgb": FilterBank.glitch_rgb,
                "anaglyph-3d": FilterBank.anaglyph_3d,
                "emboss-3d": FilterBank.emboss_3d,
            },
            "ARTISTIC": {
                "oil-paint": FilterBank.oil_paint,
                "rainbow-wave": FilterBank.rainbow_wave,
                "edge-neon": FilterBank.edge_neon,
                "pixelate": FilterBank.pixelate,
                "vhs-tape": FilterBank.vhs_tape,
                "solarize": FilterBank.solarize,
                "duotone-cyan": FilterBank.duotone_cyan,
                "cross-process": FilterBank.cross_process,
            },
            "EXOTIC": {
                "pixel-sort": FilterBank.pixel_sort,
                "kaleidoscope": FilterBank.kaleidoscope,
                "water-ripple": FilterBank.water_ripple,
                "frosted-glass": FilterBank.frosted_glass,
                "crt-screen": FilterBank.crt_screen,
                "aurora-gradient": FilterBank.aurora_gradient,
                "diamond-mosaic": FilterBank.diamond_mosaic,
                "dream-glow": FilterBank.dream_glow,
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

        self.engine = HandDetectorEngine(cfg.frame_width, cfg.frame_height, detect_scale=0.5)
        self.smoother = HandSmoother(alpha=0.45, ghost_frames=12)
        self._frame_count = 0
        self._detect_every = 2  # ponytail: skip-frame detection, smoother fills gaps

        # Naruto Ninjutsu Engines (Rasengan, Fire Style, Chidori, Shadow Clone, Blast)
        self.blast = NinjutsuBlast()
        self.rasengan = RasenganEffect(self.blast)
        self.chidori = ChidoriEffect()
        self.fireball = FireballEffect()
        self.shadow_clone = ShadowCloneEffect()
        self.last_jutsu_hand_pos = (cfg.frame_width // 2, cfg.frame_height // 2)

    def trigger_rasengan(self) -> None:
        self.rasengan.activate(*self.last_jutsu_hand_pos)
        self.toast_mgr.add("Jutsu: RASENGAN! 🌀", "J", 3.0)

    def trigger_fireball(self) -> None:
        self.fireball.activate(*self.last_jutsu_hand_pos)
        self.toast_mgr.add("Jutsu: FIRE STYLE! 🔥", "J", 3.0)

    def trigger_chidori(self) -> None:
        self.chidori.activate(*self.last_jutsu_hand_pos)
        self.toast_mgr.add("Jutsu: CHIDORI! ⚡", "J", 3.0)

    def trigger_shadow_clone(self) -> None:
        self.shadow_clone.activate()
        self.toast_mgr.add("Jutsu: SHADOW CLONE! 👥", "J", 3.0)

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)
        self.toast_mgr.add(f"Filter: {self.current_filter_name.upper()}", "*", 1.8)

    def toggle_auto_cycle(self) -> None:
        self.auto_cycle_active = not self.auto_cycle_active
        status = "ON (Every 2s)" if self.auto_cycle_active else "OFF (Gesture/Manual)"
        self.toast_mgr.add(f"Auto-Cycle: {status}", "A", 2.2)

    def cycle_theme(self) -> None:
        themes = ["ALL", "CINEMATIC", "ANIME", "CYBER", "ARTISTIC", "EXOTIC"]
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
        self.is_3d_mode = not self.is_3d_mode
        mode_label = "3D Mesh" if self.is_3d_mode else "2D Quad"
        self.toast_mgr.add(f"Mode: {mode_label}", "~", 1.8)

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

        if len(pts) == 1:
            cx, cy = pts[0]
            r = 55
            x, y = max(0, cx - r), max(0, cy - r)
            w, h = min(r * 2, frame.shape[1] - x), min(r * 2, frame.shape[0] - y)
            if w > 10 and h > 10:
                roi = frame[y : y + h, x : x + w].copy()
                processed = self.filters[filter_key](roi)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (cx - x, cy - y), r, 255, -1)
                mask_3c = cv2.merge([mask, mask, mask])
                bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
                fg = cv2.bitwise_and(processed, mask_3c)
                frame[y : y + h, x : x + w] = cv2.add(bg, fg)
                cv2.circle(frame, (cx, cy), r, (0, 240, 255), 2)
            return frame

        if len(pts) == 2:
            p1, p2 = pts[0], pts[1]
            thickness = 50
            x_min = max(0, min(p1[0], p2[0]) - thickness)
            x_max = min(frame.shape[1], max(p1[0], p2[0]) + thickness)
            y_min = max(0, min(p1[1], p2[1]) - thickness)
            y_max = min(frame.shape[0], max(p1[1], p2[1]) + thickness)
            w, h = x_max - x_min, y_max - y_min
            if w > 10 and h > 10:
                roi = frame[y_min:y_max, x_min:x_max].copy()
                processed = self.filters[filter_key](roi)
                mask = np.zeros((h, w), dtype=np.uint8)
                p1_rel = (p1[0] - x_min, p1[1] - y_min)
                p2_rel = (p2[0] - x_min, p2[1] - y_min)
                cv2.line(mask, p1_rel, p2_rel, 255, thickness)
                cv2.circle(mask, p1_rel, thickness // 2, 255, -1)
                cv2.circle(mask, p2_rel, thickness // 2, 255, -1)
                mask_3c = cv2.merge([mask, mask, mask])
                bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
                fg = cv2.bitwise_and(processed, mask_3c)
                frame[y_min:y_max, x_min:x_max] = cv2.add(bg, fg)
                cv2.line(frame, p1, p2, (0, 240, 255), 2)
            return frame

        pts_ccw = GeometryUtils.sort_polygon_vertices(pts)
        poly = np.array(pts_ccw, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(0, 240, 255), thickness=2)
        return frame

    def draw_hand_skeleton(self, frame: np.ndarray, pts: List[Tuple[int, int]]) -> None:
        if not pts or len(pts) < 21:
            return
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1], pts[p2], (255, 0, 200), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 240, 255), -1)

    def select_active_fingertips(self, hand_pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not hand_pts or len(hand_pts) < 21:
            return hand_pts if hand_pts else []

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
            return []

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
                self.toast_mgr.add(f"Filter: {self.current_filter_name.upper()}", "*", 1.8)
                self.last_auto_cycle_time = now

        # ponytail: detect every Nth frame, let velocity-predicting smoother
        # fill the gaps. Halves MediaPipe CPU cost with no visual quality loss.
        self._frame_count += 1
        if self._frame_count % self._detect_every == 0:
            raw_hands = self.engine.detect(frame)
        else:
            raw_hands = []
        hands = self.smoother.smooth(raw_hands)
        
        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if hands:
            # Clap detection triggers Rasengan 🌀
            clap_pos = HandSignClassifier.detect_clap(hands)
            if clap_pos and not self.rasengan.active:
                self.rasengan.activate(*clap_pos)
                self.toast_mgr.add("CLAP 👏 -> RASENGAN! 🌀", "J", 3.0)

            # Update primary hand position for jutsu center
            if len(hands[0]) >= 21:
                self.last_jutsu_hand_pos = (hands[0][9][0], hands[0][9][1])

            for hand_pts in hands:
                if not hand_pts:
                    continue

                if self.cfg.show_hud and len(hand_pts) >= 21:
                    self.draw_hand_skeleton(frame, hand_pts)

                selected_tips = self.select_active_fingertips(hand_pts)
                if selected_tips:
                    all_hand_tips.append(selected_tips)

                if len(hand_pts) >= 21:
                    if GeometryUtils.is_pinch_pts(hand_pts, self.cfg.pinch_threshold_ratio):
                        if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                            self.cycle_filter(1)
                            self.last_switch_time = now
                            self.pinch_anim_point = hand_pts[4]
                            self.pinch_anim_frames = 6

                    if GeometryUtils.is_fist_closed_pts(hand_pts, self.cfg.fist_dist_threshold_px):
                        fist_count += 1

                    if self.cfg.enable_gesture_snap and GeometryUtils.is_peace_sign_pts(hand_pts):
                        if now - self.last_gesture_time > self.cfg.gesture_cooldown_sec:
                            self.capture_screenshot(frame)
                            self.last_gesture_time = now

            if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
                self.toggle_mode()
                self.last_mode_toggle = now

            if len(all_hand_tips) == 2:
                t1, t2 = all_hand_tips[0], all_hand_tips[1]
                if t1 and t2:
                    frame = self.render_portal(frame, t1 + t2, self.current_filter_name)

        # --- Render Active Ninjutsu Effects (Rasengan, Fireball, Chidori, Blast, Shadow Clone) ---
        if self.rasengan.active:
            self.rasengan.update_and_draw(frame, *self.last_jutsu_hand_pos)
        if self.chidori.active:
            self.chidori.update_and_draw(frame, *self.last_jutsu_hand_pos)
        if self.fireball.active:
            self.fireball.update_and_draw(frame, *self.last_jutsu_hand_pos)
        self.blast.update_and_draw(frame)
        frame = self.shadow_clone.apply(frame)

        if self.pinch_anim_frames > 0 and self.pinch_anim_point is not None:
            r = (7 - self.pinch_anim_frames) * 6
            cv2.circle(frame, self.pinch_anim_point, r, (0, 240, 255), 2)
            self.pinch_anim_frames -= 1

        elapsed = time.time() - start_t
        if elapsed > 0:
            self.fps_tracker.append(1.0 / elapsed)
            if len(self.fps_tracker) > 20:
                self.fps_tracker.pop(0)
        current_fps = int(np.mean(self.fps_tracker)) if self.fps_tracker else 30

        if self.cfg.show_hud:
            self._draw_hud(frame, is_bowtie, current_fps)
            self.toast_mgr.render(frame)

        if self.shutter_flash_frames > 0:
            white_overlay = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, 0.4, white_overlay, 0.6, 0)
            self.shutter_flash_frames -= 1

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

        return frame

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool, fps: int) -> None:
        h, w = frame.shape[:2]
        if h < 100 or w < 300:
            return

        header_sub = frame[0:52, 0:w]
        glass_header = cv2.addWeighted(header_sub, 0.25, np.full_like(header_sub, 15), 0.75, 0)
        cv2.line(glass_header, (0, 51), (w, 51), (0, 240, 255), 1)
        frame[0:52, 0:w] = glass_header

        # Logo
        cv2.putText(frame, "HANDFLUX", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
        cv2.putText(frame, "PRO", (135, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 200), 1)

        # Mode Badge
        mode_str = "3D MESH" if self.is_3d_mode else "2D QUAD"
        cv2.rectangle(frame, (190, 14), (280, 38), (40, 40, 50), -1)
        cv2.rectangle(frame, (190, 14), (280, 38), (100, 100, 120), 1)
        cv2.putText(frame, mode_str, (198, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

        # Theme & Filter Pill
        filter_str = f"[{self.active_theme_name}] {self.current_filter_name.upper()}"
        cv2.rectangle(frame, (290, 14), (540, 38), (30, 30, 40), -1)
        cv2.rectangle(frame, (290, 14), (540, 38), (0, 240, 255), 1)
        cv2.putText(frame, filter_str, (298, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        # Auto-Cycle Pill
        if self.auto_cycle_active:
            cv2.rectangle(frame, (550, 14), (660, 38), (20, 80, 40), -1)
            cv2.rectangle(frame, (550, 14), (660, 38), (0, 255, 120), 1)
            cv2.putText(frame, "AUTO: 2s", (558, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)

        # FPS & Recording Badges
        fps_str = f"{fps} FPS"
        cv2.putText(frame, fps_str, (w - 85, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

        if self.is_recording:
            rec_dur = int(time.time() - self.rec_start_time)
            rec_str = f"REC {rec_dur//60:02d}:{rec_dur%60:02d}"
            cv2.circle(frame, (w - 175, 28), 6, (0, 0, 255), -1)
            cv2.putText(frame, rec_str, (w - 160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Glass Footer Box
        footer_sub = frame[h - 32:h, 0:w]
        glass_footer = cv2.addWeighted(footer_sub, 0.2, np.full_like(footer_sub, 10), 0.8, 0)
        cv2.line(glass_footer, (0, 0), (w, 0), (80, 80, 100), 1)
        frame[h - 32:h, 0:w] = glass_footer

        controls_str = "[A] Auto 2s | [T] Theme | [J/1] Rasengan 🌀 | [K/2] Fire 🔥 | [L/3] Chidori ⚡ | [B/4] Clone 👥 | [CLAP] 👏 | [Q] Quit"
        cv2.putText(frame, controls_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 200), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="HandFlux Pro - Real-time Hand Gesture Filter Engine")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Frame width (default: 960)")
    parser.add_argument("--height", type=int, default=540, help="Frame height (default: 540)")
    parser.add_argument("--fps", type=int, default=120, help="Target FPS (default: 120)")
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
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_index)

        if cap.isOpened():
            print(f"[INFO] Kamera terdeteksi pada indeks {cam_index}")
            break
        cap.release()

    if cap is None or not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi! Silakan hubungkan webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("HandFlux Pro Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HandFlux Pro Engine", cfg.frame_width, cfg.frame_height)
    print("[INFO] HandFlux Pro berjalan (40 Filters, 5 Themes, Naruto Ninjutsu Engine).")
    print("[INFO] Ninjutsu: [CLAP] Tepuk Tangan / [J] Rasengan | [K] Katon Fire | [L] Chidori | [B] Shadow Clone")

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
            elif key in (ord("j"), ord("J"), ord("1")):
                processor.trigger_rasengan()
            elif key in (ord("k"), ord("K"), ord("2")):
                processor.trigger_fireball()
            elif key in (ord("l"), ord("L"), ord("3")):
                processor.trigger_chidori()
            elif key in (ord("b"), ord("B"), ord("4")):
                processor.trigger_shadow_clone()
            elif key in (ord("5"),):
                processor.set_active_fingers(5)
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