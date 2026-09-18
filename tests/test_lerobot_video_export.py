import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from converters.lerobot_converter import (
    _encode_video,
    _get_ffmpeg,
    _get_latest_take_dir,
    convert_episodes,
)


class LeRobotVideoExportTests(unittest.TestCase):
    def test_export_resizes_raw_frames_and_drops_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            plain_path = os.path.join(tmp_dir, "plain.mp4")
            source_path = os.path.join(tmp_dir, "source-with-metadata.mp4")
            output_path = os.path.join(tmp_dir, "export.mp4")

            writer = cv2.VideoWriter(
                plain_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (320, 180),
            )
            self.assertTrue(writer.isOpened())
            for index in range(30):
                frame = np.full((180, 320, 3), (index * 4, 60, 200), np.uint8)
                writer.write(frame)
            writer.release()

            # Give the source a tag that must not survive the pixel-only export.
            subprocess.run(
                [
                    _get_ffmpeg(), "-y", "-loglevel", "error",
                    "-i", plain_path, "-c", "copy",
                    "-metadata", "title=SOURCE_METADATA_MUST_BE_DROPPED",
                    source_path,
                ],
                check=True,
            )

            _encode_video(source_path, output_path)

            capture = cv2.VideoCapture(output_path)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 224)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 224)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 15.0, places=2)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 15)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (224, 224, 3))

            inspection = subprocess.run(
                [_get_ffmpeg(), "-i", output_path, "-f", "null", "-"],
                capture_output=True,
                text=True,
            )
            self.assertNotIn("SOURCE_METADATA_MUST_BE_DROPPED", inspection.stderr)
            self.assertIn("Video: h264", inspection.stderr)

    def test_physical_timing_stretches_pixels_without_changing_resolution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.mp4")
            output_path = os.path.join(tmp_dir, "physical.mp4")
            writer = cv2.VideoWriter(
                source_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (320, 180),
            )
            self.assertTrue(writer.isOpened())
            for index in range(30):
                writer.write(np.full((180, 320, 3), index * 5, np.uint8))
            writer.release()

            _encode_video(source_path, output_path, time_scale=2.0)

            capture = cv2.VideoCapture(output_path)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 224)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 224)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 15.0, places=2)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
            capture.release()

    def test_rejects_invalid_time_scale(self):
        with self.assertRaises(ValueError):
            _encode_video("missing.mp4", "unused.mp4", time_scale=0)

    def test_latest_take_falls_back_from_empty_retry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            take_1 = os.path.join(tmp_dir, "takes", "take_001")
            take_2 = os.path.join(tmp_dir, "takes", "take_002")
            os.makedirs(take_1)
            os.makedirs(take_2)
            with open(os.path.join(take_1, "teach_joint.csv"), "w") as file:
                file.write("t_host_ns,q1\n")
            self.assertEqual(_get_latest_take_dir(tmp_dir), take_1)

    def test_preserved_indices_require_excluded_inputs_to_be_included(self):
        with self.assertRaisesRegex(ValueError, "include_default_excluded"):
            convert_episodes(
                [],
                "unused-dataset",
                "unused-output",
                preserve_source_indices=True,
                include_default_excluded=False,
            )


if __name__ == "__main__":
    unittest.main()
