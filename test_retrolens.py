"""
Test suite for HandFlux Pro core components
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

    def test_all_32_filters(self):
        filters = [
            # Cinematic (8)
            FilterBank.teal_orange,
            FilterBank.kodachrome,
            FilterBank.technicolor,
            FilterBank.noir_film,
            FilterBank.cinematic_warm,
            FilterBank.vignette_cinema,
            FilterBank.sepia,
            FilterBank.detail_enhance,
            # Anime & Cartoon (8)
            FilterBank.anime_cel,
            FilterBank.manga_ink,
            FilterBank.cartoon_classic,
            FilterBank.pop_art,
            FilterBank.pencil_sketch,
            FilterBank.pencil_color,
            FilterBank.stylized_water,
            FilterBank.posterize,
            # Cyber & Sci-Fi (8)
            FilterBank.cyberpunk,
            FilterBank.matrix,
            FilterBank.thermal,
            FilterBank.night_vision,
            FilterBank.hologram,
            FilterBank.glitch_rgb,
            FilterBank.anaglyph_3d,
            FilterBank.emboss_3d,
            # Artistic & EFX (8)
            FilterBank.oil_paint,
            FilterBank.rainbow_wave,
            FilterBank.edge_neon,
            FilterBank.pixelate,
            FilterBank.vhs_tape,
            FilterBank.solarize,
            FilterBank.duotone_cyan,
            FilterBank.cross_process,
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
        pts[8] = (100, 100)
        self.assertTrue(GeometryUtils.is_finger_extended(pts, "index"))

        pts[8] = (100, 350)
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
        self.assertEqual(self.processor.active_theme_name, "CINEMATIC")
        self.processor.cycle_theme()
        self.assertEqual(self.processor.active_theme_name, "ANIME")

    def test_numpy_array_pts_in_render_portal(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.array([(10, 10), (100, 10), (100, 100), (10, 100)], dtype=np.int32)
        out = self.processor.render_portal(frame, pts, "cyberpunk")
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
