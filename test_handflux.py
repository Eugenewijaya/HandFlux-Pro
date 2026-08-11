"""
Test suite for HandFlux Pro core components (v3.0.0)
"""
import os
import shutil
import unittest
import numpy as np

from HandFlux import FilterBank, GeometryUtils, PipelineConfig, PortalProcessor, ToastManager


class TestFilterBank(unittest.TestCase):
    def setUp(self):
        self.roi = np.zeros((100, 100, 3), dtype=np.uint8)
        self.roi[25:75, 25:75] = [200, 150, 100]

    def test_all_18_special_effects_filters(self):
        cfg = PipelineConfig(frame_width=640, frame_height=480)
        processor = PortalProcessor(cfg)
        filters = list(processor.filters.values())
        self.assertGreaterEqual(len(filters), 18)
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

    def test_make_flexible_quad(self):
        h1 = [(0, 0)] * 21
        h2 = [(0, 0)] * 21
        h1[8] = (10, 10)    # Left Index
        h1[4] = (10, 100)   # Left Thumb
        h2[4] = (100, 100)  # Right Thumb
        h2[8] = (100, 10)   # Right Index
        quad = GeometryUtils.make_flexible_quad(h1, h2)
        self.assertEqual(len(quad), 4)

    def test_is_finger_extended(self):
        pts = [(0, 0)] * 21
        pts[0] = (100, 500)  # Wrist
        pts[5] = (100, 400)  # Index MCP
        pts[6] = (100, 300)  # Index PIP
        pts[8] = (100, 100)  # Index Tip (Extended)
        self.assertTrue(GeometryUtils.is_finger_extended(pts, "index"))

        pts[8] = (100, 450)  # Index Tip Curled into palm
        self.assertFalse(GeometryUtils.is_finger_extended(pts, "index"))


class TestPortalProcessor(unittest.TestCase):
    def setUp(self):
        self.out_dir = "test_captures"
        self.cfg = PipelineConfig(frame_width=640, frame_height=480, output_dir=self.out_dir)
        self.processor = PortalProcessor(self.cfg)

    def tearDown(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_auto_cycle_toggle(self):
        self.assertFalse(self.processor.auto_cycle_active)
        self.processor.toggle_auto_cycle()
        self.assertTrue(self.processor.auto_cycle_active)

    def test_theme_switcher(self):
        self.assertEqual(self.processor.active_theme_name, "ALL")
        self.processor.cycle_theme()
        self.assertEqual(self.processor.active_theme_name, "Y2K POP-ART")
        self.processor.cycle_theme()
        self.assertEqual(self.processor.active_theme_name, "SPECIAL FX")

    def test_numpy_array_pts_in_render_portal(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.array([(10, 10), (100, 10), (100, 100), (10, 100)], dtype=np.int32)
        out = self.processor.render_portal(frame, pts, self.processor.current_filter_name)
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()

