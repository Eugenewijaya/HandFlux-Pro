"""
Test suite for RetroLens core components
"""
import os
import shutil
import unittest
import numpy as np

from Retrolens import FilterBank, GeometryUtils, PipelineConfig, PortalProcessor, ToastManager


class TestFilterBank(unittest.TestCase):
    def setUp(self):
        self.roi = np.zeros((100, 100, 3), dtype=np.uint8)
        self.roi[25:75, 25:75] = [200, 150, 100]

    def test_all_filters(self):
        filters = [
            FilterBank.dual_tone,
            FilterBank.thermal,
            FilterBank.sketch,
            FilterBank.pixelate,
            FilterBank.glitch,
            FilterBank.invert,
            FilterBank.red_channel,
            FilterBank.edge,
            FilterBank.blur,
            FilterBank.cartoon,
            FilterBank.rainbow_wave,
            FilterBank.cyberpunk,
            FilterBank.vhs,
            FilterBank.matrix,
            FilterBank.pop_art,
            FilterBank.sepia,
        ]
        for f in filters:
            out = f(self.roi)
            self.assertEqual(out.shape, self.roi.shape, f"Filter {f.__name__} returned wrong shape")


class TestGeometryUtils(unittest.TestCase):
    def test_euclidean_dist(self):
        dist = GeometryUtils.euclidean_dist((0, 0), (3, 4))
        self.assertAlmostEqual(dist, 5.0)

    def test_sort_polygon_vertices(self):
        pts = [(10, 10), (100, 10), (100, 100), (10, 100)]
        sorted_pts = GeometryUtils.sort_polygon_vertices(pts)
        self.assertEqual(len(sorted_pts), 4)

    def test_is_finger_extended(self):
        pts = [(0, 0)] * 21
        pts[0] = (100, 500)
        pts[6] = (100, 300)
        pts[8] = (100, 100)  # Index tip far from wrist
        self.assertTrue(GeometryUtils.is_finger_extended(pts, "index"))

        pts[8] = (100, 350)  # Index tip folded below PIP joint
        self.assertFalse(GeometryUtils.is_finger_extended(pts, "index"))


class TestPortalProcessor(unittest.TestCase):
    def setUp(self):
        self.out_dir = "test_captures"
        self.cfg = PipelineConfig(frame_width=640, frame_height=480, output_dir=self.out_dir)
        self.processor = PortalProcessor(self.cfg)

    def tearDown(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_numpy_array_pts_in_render_portal(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.array([(10, 10), (100, 10), (100, 100), (10, 100)], dtype=np.int32)
        out = self.processor.render_portal(frame, pts, "cyberpunk")
        self.assertEqual(out.shape, (480, 640, 3))

    def test_gesture_snap_toggle(self):
        self.assertFalse(self.processor.cfg.enable_gesture_snap)
        self.processor.toggle_gesture_snap()
        self.assertTrue(self.processor.cfg.enable_gesture_snap)

    def test_finger_setting(self):
        self.assertEqual(self.processor.cfg.active_fingers, 5)
        self.processor.set_active_fingers(3)
        self.assertEqual(self.processor.cfg.active_fingers, 3)

    def test_screenshot_capture(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        filepath = self.processor.capture_screenshot(frame)
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()
