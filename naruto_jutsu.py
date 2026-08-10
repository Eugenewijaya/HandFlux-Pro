"""
Naruto Jutsu Camera - Standalone Ninjutsu Visual Effects Engine
Powered by HandFlux Technology & Kaggle Naruto Hand Gesture Model
"""

import argparse
from dataclasses import dataclass, field
import math
import os
import random
import sys
import time
from typing import List, Tuple, Optional

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
            import urllib.request
            print("[INFO] Mengunduh model deteksi tangan MediaPipe (hand_landmarker.task)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[INFO] Model berhasil diunduh!")
            return True
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh model: {e}")
            return False
    return True


class HandSmoother:
    def __init__(self, alpha: float = 0.45, ghost_frames: int = 12) -> None:
        self.alpha = alpha
        self.ghost_frames = ghost_frames
        self.prev_hands: List[List[Tuple[float, float]]] = []
        self.velocities: List[List[Tuple[float, float]]] = []
        self._miss_count: int = 0

    @staticmethod
    def _centroid(hand: list) -> Tuple[float, float]:
        xs = [p[0] for p in hand]
        ys = [p[1] for p in hand]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _match_hands(self, new_hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if len(self.prev_hands) < 2 or len(new_hands) < 2:
            return new_hands
        prev_c = [self._centroid(h) for h in self.prev_hands]
        new_c = [self._centroid(h) for h in new_hands]
        d_same = sum(np.hypot(prev_c[i][0] - new_c[i][0], prev_c[i][1] - new_c[i][1]) for i in range(2))
        d_swap = sum(np.hypot(prev_c[i][0] - new_c[1 - i][0], prev_c[i][1] - new_c[1 - i][1]) for i in range(2))
        if d_swap < d_same:
            return [new_hands[1], new_hands[0]]
        return new_hands

    def smooth(self, hands: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        if not hands:
            if self.prev_hands and self._miss_count < self.ghost_frames:
                self._miss_count += 1
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
    def __init__(self, frame_width: int, frame_height: int, detect_scale: float = 0.5) -> None:
        self.w = frame_width
        self.h = frame_height
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

    def detect(self, frame: np.ndarray) -> List[List[Tuple[int, int]]]:
        if frame is None or frame.size == 0:
            return []

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

        return hands_list


# ==============================================================================
#  NINJUTSU VISUAL EFFECT CLASSES
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


class LoveBalloon:
    def __init__(self, w: int, h: int) -> None:
        self.x = float(random.randint(20, max(20, w - 20)))
        self.y = float(h + random.randint(10, 60))
        self.size = float(random.randint(18, 38))
        self.speed_y = float(random.uniform(2.5, 5.2))
        self.wobble_freq = float(random.uniform(2.0, 5.0))
        self.wobble_amp = float(random.uniform(1.2, 3.5))
        self.start_t = time.time()
        self.color = random.choice([
            (180, 105, 255),  # Soft Pink (BGR)
            (80, 40, 240),    # Romantic Crimson
            (220, 60, 255),   # Vibrant Magenta
            (100, 50, 255),   # Deep Rose
            (240, 240, 255),  # Pure White
            (80, 215, 255),   # Warm Peach
        ])
        self.alpha = float(random.uniform(0.78, 0.95))

    def update(self) -> None:
        self.y -= self.speed_y
        elapsed = time.time() - self.start_t
        self.x += math.sin(elapsed * self.wobble_freq) * self.wobble_amp

    def is_alive(self) -> bool:
        return self.y > -80

    def draw(self, frame: np.ndarray) -> None:
        if not self.is_alive():
            return
        h, w = frame.shape[:2]
        cx, cy = int(self.x), int(self.y)
        sz = self.size

        if cx < -50 or cx > w + 50 or cy < -50 or cy > h + 80:
            return

        top_alpha = min(1.0, max(0.0, (cy + 20) / 120.0)) * self.alpha

        pts = []
        num_pts = 30
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
        cv2.circle(ov, (hx_shine, hy_shine), max(2, int(sz * 0.14)), (255, 255, 255), -1)

        s_pts = []
        for s in range(10):
            sx = int(cx + math.sin(s * 0.6) * 3)
            sy = int(cy + sz * 0.8 + s * 3.5)
            s_pts.append((sx, sy))
        cv2.polylines(ov, [np.array(s_pts, dtype=np.int32)], False, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.addWeighted(ov, top_alpha, frame, 1.0 - top_alpha, 0, frame)


class LoveBalloonEngine:
    def __init__(self) -> None:
        self.balloons: List[LoveBalloon] = []

    def spawn(self, w: int, h: int, count: int = 3) -> None:
        for _ in range(count):
            self.balloons.append(LoveBalloon(w, h))

    def update_and_draw(self, frame: np.ndarray) -> None:
        self.balloons = [b for b in self.balloons if b.is_alive()]
        for b in self.balloons:
            b.update()
            b.draw(frame)


def is_v_sign(pts: List[Tuple[int, int]]) -> bool:
    if not pts or len(pts) < 21:
        return False
    wrist = np.array(pts[0])
    d_index = np.linalg.norm(np.array(pts[8]) - wrist)
    d_middle = np.linalg.norm(np.array(pts[12]) - wrist)
    d_ring = np.linalg.norm(np.array(pts[16]) - wrist)
    d_pinky = np.linalg.norm(np.array(pts[20]) - wrist)
    return (d_index > d_ring * 1.2) and (d_middle > d_pinky * 1.2) and (d_ring < d_index * 0.75)


def main() -> None:
    w, h = 960, 540
    engine = HandDetectorEngine(w, h, detect_scale=0.5)
    smoother = HandSmoother(alpha=0.45, ghost_frames=12)

    blast = NinjutsuBlast()
    rasengan = RasenganEffect(blast)
    chidori = ChidoriEffect()
    fireball = FireballEffect()
    shadow_clone = ShadowCloneEffect()

    love_engine = LoveBalloonEngine()
    v_blur_timer = 0.0
    v_blur_duration = 3.5
    v_blur_dir = "foto kita blurr"
    last_v_snap_time = 0.0
    os.makedirs(v_blur_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 120)

    cv2.namedWindow("Naruto Ninjutsu Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Naruto Ninjutsu Camera", w, h)

    print("\n🍃 NARUTO NINJUTSU CAMERA ENGINE")
    print("  👏 Tepuk Tangan (CLAP) atau Tekan '1' / 'J'  -> RASENGAN + BLAST")
    print("  🔥 Tekan '2' / 'K'                           -> FIRE STYLE (KATON)")
    print("  ⚡ Tekan '3' / 'L'                           -> CHIDORI")
    print("  👥 Tekan '4' / 'B'                           -> SHADOW CLONE")
    print("  ✌️  Gesture 'V'                              -> LOVE BLUR 40% + BALLOONS")
    print("  ❌ Tekan 'Q' atau ESC                       -> Keluar\n")

    frame_count = 0
    hand_pos = (w // 2, h // 2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()

            frame_count += 1
            if frame_count % 2 == 0:
                raw_hands = engine.detect(frame)
            else:
                raw_hands = []

            hands = smoother.smooth(raw_hands)

            if hands:
                p1 = hands[0][9]
                if len(hands) >= 2:
                    p2 = hands[1][9]
                    dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                    if dist < 90 and not rasengan.active:
                        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                        rasengan.activate(cx, cy)
                hand_pos = (p1[0], p1[1])

                for hand_pts in hands:
                    for p1_idx, p2_idx in HAND_CONNECTIONS:
                        cv2.line(frame, hand_pts[p1_idx], hand_pts[p2_idx], (255, 0, 200), 2)
                    for pt in hand_pts:
                        cv2.circle(frame, pt, 4, (0, 240, 255), -1)

                    if is_v_sign(hand_pts):
                        v_blur_timer = now + v_blur_duration
                        if now - last_v_snap_time > 2.5:
                            fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                            cv2.imwrite(fn, frame)
                            print(f"[INFO] Foto Love Blur disimpan ke: {os.path.abspath(fn)}")
                            last_v_snap_time = now

            if rasengan.active:
                rasengan.update_and_draw(frame, *hand_pos)
            if chidori.active:
                chidori.update_and_draw(frame, *hand_pos)
            if fireball.active:
                fireball.update_and_draw(frame, *hand_pos)
            blast.update_and_draw(frame)
            frame = shadow_clone.apply(frame)

            # Apply V-Sign 40% Blur + Floating Love Balloons
            if now < v_blur_timer:
                blurred = cv2.GaussianBlur(frame, (29, 29), 0)
                frame = cv2.addWeighted(frame, 0.60, blurred, 0.40, 0)
                love_engine.spawn(w, h, count=3)
                love_engine.update_and_draw(frame)

                cv2.rectangle(frame, (w // 2 - 170, 10), (w // 2 + 170, 42), (40, 20, 60), -1)
                cv2.rectangle(frame, (w // 2 - 170, 10), (w // 2 + 170, 42), (220, 100, 255), 1)
                cv2.putText(frame, "LOVE BLUR 40% (foto kita blurr)", (w // 2 - 155, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 255), 1, cv2.LINE_AA)

            # Draw HUD legend
            cv2.putText(frame, "NARUTO NINJUTSU ENGINE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
            cv2.putText(frame, "[CLAP/1] Rasengan  [2] Katon Fire  [3] Chidori  [4] Clone  [V-Sign] Love Blur  [Q] Quit", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow("Naruto Ninjutsu Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("1"), ord("j"), ord("J")):
                rasengan.activate(*hand_pos)
            elif key in (ord("2"), ord("k"), ord("K")):
                fireball.activate(*hand_pos)
            elif key in (ord("3"), ord("l"), ord("L")):
                chidori.activate(*hand_pos)
            elif key in (ord("4"), ord("b"), ord("B")):
                shadow_clone.activate()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
