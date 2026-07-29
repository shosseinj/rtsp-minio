import unittest

from app.ai.coordinate_transform import BBox, source_to_target


class CoordinateTransformTests(unittest.TestCase):
    def test_reference_identity(self):
        box = BBox(192, 108, 384, 432)
        self.assertEqual(source_to_target(box, 1920, 1080, 1920, 1080), box)

    def test_reference_720p(self):
        actual = source_to_target(
            BBox(192, 108, 384, 432), 1920, 1080, 1280, 720
        )
        self.assertEqual(actual, BBox(128, 72, 256, 288))

    def test_supported_landscape_and_portrait_shapes(self):
        for width, height in (
            (1920, 1080), (2560, 1440), (2944, 1664),
            (1280, 720), (1080, 1920),
        ):
            box = BBox(width * .1, height * .1, width * .2, height * .4)
            actual = source_to_target(box, width, height, 1280, 720)
            self.assertGreaterEqual(actual.x, 0)
            self.assertGreaterEqual(actual.y, 0)
            self.assertLessEqual(actual.x + actual.width, 1280)
            self.assertLessEqual(actual.y + actual.height, 720)


if __name__ == "__main__":
    unittest.main()
