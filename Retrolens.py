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
    """Exponential Moving Average (EMA) position smoother to eliminate landmark jitter."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.prev_hands: List[List[Tuple[float, float]]] = []

    def smooth(self, hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if not hands:
            self.prev_hands = []
            return []

        smoothed_hands = []
        for i, hand in enumerate(hands):
            if not hand:
                continue
            if i < len(self.prev_hands) and len(self.prev_hands[i]) == len(hand):
                prev = self.prev_hands[i]
                smoothed = [
                    (
                        int(self.alpha * curr[0] + (1 - self.alpha) * prev[j][0]),
                        int(self.alpha * curr[1] + (1 - self.alpha) * prev[j][1]),
                    )
                    for j, curr in enumerate(hand)
                ]
            else:
                smoothed = hand

            smoothed_hands.append(smoothed)

        self.prev_hands = [[(float(p[0]), float(p[1])) for p in h] for h in smoothed_hands]
        return smoothed_hands


class HandDetectorEngine:
    """Multi-backend hand tracking engine supporting MediaPipe Tasks, MediaPipe Solutions, and Fallback."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height
        self.mode = "none"
        self.tasks_detector = None
        self.solutions_detector = None
        self.mp_module = None

        try:
            if ensure_model_exists() and os.path.exists(MODEL_PATH):
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
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
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
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
    """Collection of 25 real-time video filters categorized into Themes."""

    # --- CINEMATIC THEME ---
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

    # --- ANIME & CARTOON THEME ---
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
    def posterize(roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            return roi
        n_colors = 4
        return np.clip((roi // (256 // n_colors)) * (256 // n_colors), 0, 255).astype(np.uint8)

    # --- CYBER & SCI-FI THEME ---
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

    # --- ARTISTIC & EFX THEME ---
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

        return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist)) * 1.08

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

        # 25 Filters mapped by Theme
        self.themed_filters: Dict[str, Dict[str, Callable[[np.ndarray], np.ndarray]]] = {
            "CINEMATIC": {
                "teal-orange": FilterBank.teal_orange,
                "kodachrome": FilterBank.kodachrome,
                "technicolor": FilterBank.technicolor,
                "noir-film": FilterBank.noir_film,
                "cinematic-warm": FilterBank.cinematic_warm,
                "vignette-cinema": FilterBank.vignette_cinema,
                "sepia-vintage": FilterBank.sepia,
            },
            "ANIME": {
                "anime-cel": FilterBank.anime_cel,
                "manga-ink": FilterBank.manga_ink,
                "cartoon-classic": FilterBank.cartoon_classic,
                "pop-art": FilterBank.pop_art,
                "pencil-sketch": FilterBank.pencil_sketch,
                "posterize": FilterBank.posterize,
            },
            "CYBER": {
                "cyberpunk": FilterBank.cyberpunk,
                "matrix": FilterBank.matrix,
                "thermal": FilterBank.thermal,
                "night-vision": FilterBank.night_vision,
                "hologram": FilterBank.hologram,
                "glitch-rgb": FilterBank.glitch_rgb,
            },
            "ARTISTIC": {
                "oil-paint": FilterBank.oil_paint,
                "rainbow-wave": FilterBank.rainbow_wave,
                "edge-neon": FilterBank.edge_neon,
                "pixelate": FilterBank.pixelate,
                "vhs-tape": FilterBank.vhs_tape,
                "solarize": FilterBank.solarize,
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

        self.engine = HandDetectorEngine(cfg.frame_width, cfg.frame_height)
        self.smoother = HandSmoother(alpha=0.5)

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
        themes = ["ALL", "CINEMATIC", "ANIME", "CYBER", "ARTISTIC"]
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

        # Handle 1-finger (circle portal)
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

        # Handle 2-finger (pill/capsule portal)
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

        # Handle 3+ fingers (polygon portal sorted CCW around centroid - NEVER FOLDS)
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
        frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        now = time.time()

        # Handle Auto-Cycle Filter (Switch every 2 seconds if active)
        if self.auto_cycle_active:
            if now - self.last_auto_cycle_time > self.cfg.auto_cycle_interval:
                self.cycle_filter(1)
                self.last_auto_cycle_time = now

        raw_hands = self.engine.detect(frame)
        hands = self.smoother.smooth(raw_hands)
        
        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if hands:
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

            # Portal ONLY opens when TWO hands are present
            if len(all_hand_tips) == 2:
                t1, t2 = all_hand_tips[0], all_hand_tips[1]
                if t1 and t2:
                    frame = self.render_portal(frame, t1 + t2, self.current_filter_name)

        # Draw pinch ripple effect
        if self.pinch_anim_frames > 0 and self.pinch_anim_point is not None:
            r = (7 - self.pinch_anim_frames) * 6
            cv2.circle(frame, self.pinch_anim_point, r, (0, 240, 255), 2)
            self.pinch_anim_frames -= 1

        # Track FPS
        elapsed = time.time() - start_t
        if elapsed > 0:
            self.fps_tracker.append(1.0 / elapsed)
            if len(self.fps_tracker) > 20:
                self.fps_tracker.pop(0)
        current_fps = int(np.mean(self.fps_tracker)) if self.fps_tracker else 30

        # Draw HUD overlays if enabled
        if self.cfg.show_hud:
            self._draw_hud(frame, is_bowtie, current_fps)
            self.toast_mgr.render(frame)

        # Handle Shutter Flash
        if self.shutter_flash_frames > 0:
            white_overlay = np.full_like(frame, 255)
            frame = cv2.addWeighted(frame, 0.4, white_overlay, 0.6, 0)
            self.shutter_flash_frames -= 1

        # Record frame if active
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

        # FPS & Recording Badges (Right side)
        fps_str = f"{fps} FPS"
        cv2.putText(frame, fps_str, (w - 85, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

        if self.is_recording:
            rec_dur = int(time.time() - self.rec_start_time)
            rec_str = f"REC {rec_dur//60:02d}:{rec_dur%60:02d}"
            cv2.circle(frame, (w - 175, 28), 6, (0, 0, 255), -1)
            cv2.putText(frame, rec_str, (w - 160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Glass Footer Box (Bottom bar)
        footer_sub = frame[h - 32:h, 0:w]
        glass_footer = cv2.addWeighted(footer_sub, 0.2, np.full_like(footer_sub, 10), 0.8, 0)
        cv2.line(glass_footer, (0, 0), (w, 0), (80, 80, 100), 1)
        frame[h - 32:h, 0:w] = glass_footer

        controls_str = "[A] Auto 2s | [T] Theme | [1-5/F] Fingers | [S] Snap | [N/P] Filter | [R] Rec | [Q] Quit"
        cv2.putText(frame, controls_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 200), 1)


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
    print("[INFO] HandFlux Pro berjalan (25 Filters, Auto-Cycle Mode Available).")
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