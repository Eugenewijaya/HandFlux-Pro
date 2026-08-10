#!/usr/bin/env python3
"""
Foto Kita Blur Engine - Real-time Romantic 40% Love Blur Camera for Couples
Detects V-Sign (Peace Sign ✌️) hand gesture to activate smooth 40% Screen Blur & Floating Love Balloons.
Includes real-time HD Camera Enhancement (CLAHE + Unsharp Masking + Warm Skin Tone) & Couple Aesthetic UI.
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


def enhance_camera_hd(frame: np.ndarray) -> np.ndarray:
    """
    Applies real-time HD Camera Enhancement:
    1. LAB CLAHE for adaptive contrast & detail expansion
    2. Unsharp Masking for crisp HD sharpness
    3. Gentle warm skin-tone tint for flattering couple photos
    """
    if frame is None or frame.size == 0:
        return frame

    # 1. LAB CLAHE for contrast and facial detail enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 2. Unsharp Masking for crisp HD clarity
    gaussian = cv2.GaussianBlur(bgr, (0, 0), 2.2)
    hd_frame = cv2.addWeighted(bgr, 1.20, gaussian, -0.20, 0)

    # 3. Flattering warm skin tint (+5 Red channel boost)
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
    def __init__(self, w: int, h: int) -> None:
        self.x = float(random.randint(20, max(20, w - 20)))
        self.y = float(h + random.randint(10, 60))
        self.size = float(random.randint(22, 42))
        self.speed_y = float(random.uniform(2.2, 4.8))
        self.wobble_freq = float(random.uniform(2.0, 4.5))
        self.wobble_amp = float(random.uniform(1.2, 3.2))
        self.start_t = time.time()
        self.color = random.choice([
            (180, 105, 255),  # Soft Pink (BGR)
            (90, 50, 245),    # Romantic Crimson
            (220, 90, 255),   # Vibrant Rose
            (120, 60, 255),   # Deep Magenta
            (245, 245, 255),  # Pearl White
            (100, 200, 255),  # Soft Gold / Coral
        ])
        self.alpha = float(random.uniform(0.82, 0.96))

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

        # Smooth parametric heart balloon
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

        # White shiny reflection highlight
        hx_shine = int(cx - sz * 0.25)
        hy_shine = int(cy - sz * 0.35)
        cv2.circle(ov, (hx_shine, hy_shine), max(2, int(sz * 0.15)), (255, 255, 255), -1)

        # Curved string tail
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


def is_v_sign(pts: List[Tuple[int, int]]) -> bool:
    if not pts or len(pts) < 21:
        return False
    wrist = np.array(pts[0])
    d_index = np.linalg.norm(np.array(pts[8]) - wrist)
    d_middle = np.linalg.norm(np.array(pts[12]) - wrist)
    d_ring = np.linalg.norm(np.array(pts[16]) - wrist)
    d_pinky = np.linalg.norm(np.array(pts[20]) - wrist)
    return (d_index > d_ring * 1.2) and (d_middle > d_pinky * 1.2) and (d_ring < d_index * 0.75)


def draw_romantic_ui(frame: np.ndarray, blur_weight: float, toast_msg: str, toast_time: float) -> None:
    """
    Renders an elegant, romantic glassmorphic UI tailored for couples.
    No tech debug lines or raw numbers.
    """
    h, w = frame.shape[:2]

    # Glass Header Pill (Rose Gold Theme)
    header_w = 260
    header_h = 44
    hx1 = (w - header_w) // 2
    hy1 = 12
    hx2 = hx1 + header_w
    hy2 = hy1 + header_h

    sub_hdr = frame[hy1:hy2, hx1:hx2]
    if sub_hdr.size > 0:
        glass = cv2.addWeighted(sub_hdr, 0.35, np.full_like(sub_hdr, (40, 20, 60)), 0.65, 0)
        cv2.rectangle(glass, (0, 0), (header_w, header_h), (200, 140, 255), 1)
        frame[hy1:hy2, hx1:hx2] = glass
        cv2.putText(frame, "Foto Kita  ", (hx1 + 45, hy1 + 28), cv2.FONT_HERSHEY_TRIPLEX, 0.65, (255, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOVE BLUR", (hx1 + 145, hy1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 105, 255), 1, cv2.LINE_AA)

    # Toast Notification
    now = time.time()
    if toast_msg and (now - toast_time < 3.0):
        alpha_t = min(1.0, max(0.0, (3.0 - (now - toast_time)) / 0.5))
        (tw, th), _ = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1 = (w - tw) // 2 - 15
        ty1 = hy2 + 15
        tx2 = tx1 + tw + 30
        ty2 = ty1 + th + 14

        if tx1 > 0 and tx2 < w and ty1 > 0 and ty2 < h:
            sub_t = frame[ty1:ty2, tx1:tx2]
            glass_t = cv2.addWeighted(sub_t, 0.25, np.full_like(sub_t, (50, 25, 70)), 0.75, 0)
            cv2.rectangle(glass_t, (0, 0), (tx2 - tx1, ty2 - ty1), (255, 160, 220), 1)
            frame[ty1:ty2, tx1:tx2] = glass_t
            cv2.putText(frame, toast_msg, (tx1 + 15, ty1 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 240, 255), 1, cv2.LINE_AA)

    # Romantic Footer Guide
    footer_w = 420
    footer_h = 32
    fx1 = (w - footer_w) // 2
    fy1 = h - 42
    fx2 = fx1 + footer_w
    fy2 = fy1 + footer_h

    sub_ftr = frame[fy1:fy2, fx1:fx2]
    if sub_ftr.size > 0:
        glass_ftr = cv2.addWeighted(sub_ftr, 0.25, np.full_like(sub_ftr, (30, 15, 45)), 0.75, 0)
        cv2.rectangle(glass_ftr, (0, 0), (footer_w, footer_h), (180, 120, 220), 1)
        frame[fy1:fy2, fx1:fx2] = glass_ftr
        cv2.putText(frame, "Pose Peace (V) Bersama Pasangan untuk Blur 40%", (fx1 + 18, fy1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 210, 255), 1, cv2.LINE_AA)


def main() -> None:
    w, h = 960, 540
    engine = HandDetectorEngine(w, h, detect_scale=0.5)
    smoother = HandSmoother(alpha=0.45, ghost_frames=12)
    love_engine = LoveBalloonEngine()

    v_blur_dir = "foto kita blurr"
    os.makedirs(v_blur_dir, exist_ok=True)

    # Smooth transition state variable (0.0 = Normal HD, 1.0 = Full 40% Love Blur)
    blur_weight = 0.0
    target_blur_weight = 0.0
    v_sign_active_until = 0.0
    last_v_snap_time = 0.0
    shutter_flash_frames = 0
    toast_msg = ""
    toast_time = 0.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 120)

    cv2.namedWindow("Foto Kita Blur Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foto Kita Blur Camera", w, h)

    print("\n❤️ FOTO KITA BLUR — CAMERA ENGINE (COUPLE EDITION)")
    print("  ✌️  Gestur Tangan Peace (V)  -> Smooth 40% Screen Blur + Floating Love Balloons")
    print("  ✨  HD Camera Enhancer      -> Active (LAB CLAHE + Unsharp Masking)")
    print("  📸  Foto otomatis disimpan ke -> 'foto kita blurr/'")
    print("  📸  Tekan 'S'                -> Foto manual")
    print("  ❌  Tekan 'Q' atau ESC      -> Keluar\n")

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

            v_sign_detected = False
            if hands:
                for hand_pts in hands:
                    if is_v_sign(hand_pts):
                        v_sign_detected = True
                        break

            # Trigger 40% Love Blur on V-Sign
            if v_sign_detected:
                v_sign_active_until = now + 3.2
                if now - last_v_snap_time > 2.5:
                    fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                    cv2.imwrite(fn, frame)
                    print(f"[INFO] Foto Love Blur disimpan ke: {os.path.abspath(fn)}")
                    last_v_snap_time = now
                    shutter_flash_frames = 4
                    toast_msg = "Foto tersimpan di 'foto kita blurr'! ♡"
                    toast_time = now

            target_blur_weight = 1.0 if now < v_sign_active_until else 0.0

            # 2. Smooth Lerp Transition between Normal & 40% Blur (No harsh snapping!)
            lerp_speed = 0.08
            blur_weight += (target_blur_weight - blur_weight) * lerp_speed

            # 3. Apply Smooth 40% Blur Blend when active
            if blur_weight > 0.01:
                blurred = cv2.GaussianBlur(frame, (31, 31), 0)
                effective_alpha = blur_weight * 0.40  # Exactly 40% max blur
                frame = cv2.addWeighted(frame, 1.0 - effective_alpha, blurred, effective_alpha, 0)

                # Spawn and render floating love balloons
                love_engine.spawn(w, h, count=2)
                love_engine.update_and_draw(frame, global_alpha=blur_weight)

            # 4. Render Romantic Couple Glassmorphic UI
            draw_romantic_ui(frame, blur_weight, toast_msg, toast_time)

            # Camera Shutter Flash Animation
            if shutter_flash_frames > 0:
                white_overlay = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 0.45, white_overlay, 0.55, 0)
                shutter_flash_frames -= 1

            cv2.imshow("Foto Kita Blur Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                fn = os.path.join(v_blur_dir, f"foto_kita_blurr_{int(time.time() * 1000)}.png")
                cv2.imwrite(fn, frame)
                shutter_flash_frames = 4
                toast_msg = "Foto tersimpan manual! ♡"
                toast_time = now
                print(f"[INFO] Foto disimpan manual ke: {os.path.abspath(fn)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
