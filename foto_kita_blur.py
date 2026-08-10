#!/usr/bin/env python3
"""
Foto Kita Blur Engine - Real-time Romantic 40% Love Blur Camera
Detects V-Sign (Peace Sign ✌️) hand gesture to activate 40% Screen Blur & Floating Love Balloons.
Saves all captures locally to 'foto kita blurr/' folder.
"""

import math
import os
import random
import sys
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
    def __init__(self, w: int, h: int) -> None:
        self.x = float(random.randint(20, max(20, w - 20)))
        self.y = float(h + random.randint(10, 60))
        self.size = float(random.randint(18, 38))
        self.speed_y = float(random.uniform(2.5, 5.2))
        self.wobble_freq = float(random.uniform(2.0, 5.0))
        self.wobble_amp = float(random.uniform(1.2, 3.5))
        self.start_t = time.time()
        self.color = random.choice([
            (180, 105, 255),  # Soft Pink
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

    cv2.namedWindow("Foto Kita Blur Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foto Kita Blur Camera", w, h)

    print("\n❤️ FOTO KITA BLUR CAMERA ENGINE")
    print("  ✌️  Gestur Tangan V (Peace Sign) -> Activate 40% Screen Blur + Floating Love Balloons")
    print("  📸  Foto otomatis disimpan ke  -> 'foto kita blurr/'")
    print("  📸  Tekan 'S'                   -> Take manual photo")
    print("  ❌  Tekan 'Q' atau ESC         -> Keluar\n")

    frame_count = 0

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
                for hand_pts in hands:
                    for p1_idx, p2_idx in HAND_CONNECTIONS:
                        cv2.line(frame, hand_pts[p1_idx], hand_pts[p2_idx], (255, 105, 180), 2)
                    for pt in hand_pts:
                        cv2.circle(frame, pt, 4, (0, 240, 255), -1)

                    if is_v_sign(hand_pts):
                        v_blur_timer = now + v_blur_duration
                        if now - last_v_snap_time > 2.5:
                            fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                            cv2.imwrite(fn, frame)
                            print(f"[INFO] Foto Love Blur disimpan ke: {os.path.abspath(fn)}")
                            last_v_snap_time = now

            # Render 40% Gaussian Blur + Floating Love Balloons
            if now < v_blur_timer:
                blurred = cv2.GaussianBlur(frame, (29, 29), 0)
                frame = cv2.addWeighted(frame, 0.60, blurred, 0.40, 0)
                love_engine.spawn(w, h, count=3)
                love_engine.update_and_draw(frame)

                cv2.rectangle(frame, (w // 2 - 180, 10), (w // 2 + 180, 42), (40, 20, 60), -1)
                cv2.rectangle(frame, (w // 2 - 180, 10), (w // 2 + 180, 42), (220, 100, 255), 1)
                cv2.putText(frame, "LOVE BLUR 40% (foto kita blurr)", (w // 2 - 165, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 255), 1, cv2.LINE_AA)

            # Draw HUD Legend
            cv2.putText(frame, "FOTO KITA BLUR", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 105, 180), 2)
            cv2.putText(frame, "[✌️ V-Sign] 40% Love Blur & Balloons  [S] Save Photo  [Q] Quit", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            cv2.imshow("Foto Kita Blur Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                cv2.imwrite(fn, frame)
                print(f"[INFO] Foto disimpan manual ke: {os.path.abspath(fn)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
