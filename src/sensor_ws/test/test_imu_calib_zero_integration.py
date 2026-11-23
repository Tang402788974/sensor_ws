#!/usr/bin/env python3
"""Black-box tests for the zero-calibration ROS node; run in a sourced ROS shell."""

import pathlib
import json
import subprocess
import sys
import tempfile
import time
import unittest


def package_prefix():
    return pathlib.Path(
        subprocess.check_output(["ros2", "pkg", "prefix", "sensor_ws"], text=True).strip()
    )


class ZeroCalibrationIntegrationTest(unittest.TestCase):
    def run_calibration(self, invalid_every_n, unknown_imu_id=0, min_complete_samples=1):
        prefix = package_prefix()
        publisher_path = prefix / "lib/sensor_ws/imu_test_publisher.py"
        node_path = prefix / "lib/sensor_ws/imu_calib_zero"

        with tempfile.TemporaryDirectory(prefix="imu-calibration-test-") as directory:
            calibration_file = pathlib.Path(directory) / "calibration.json"
            calibration_file.write_text("preserve-existing-calibration", encoding="utf-8")
            publisher = subprocess.Popen(
                [
                    sys.executable,
                    str(publisher_path),
                    "--ros-args",
                    "-p",
                    "invalid_imu_id:=4",
                    "-p",
                    f"invalid_every_n:={invalid_every_n}",
                    "-p",
                    f"unknown_imu_id:={unknown_imu_id}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.5)
                result = subprocess.run(
                    [
                        str(node_path),
                        "--ros-args",
                        "-p",
                        "duration_sec:=0.5",
                        "-p",
                        f"min_complete_samples:={min_complete_samples}",
                        "-p",
                        f"calibration_file:={calibration_file}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
            finally:
                publisher.terminate()
                publisher.wait(timeout=5)

            return result, calibration_file.read_text(encoding="utf-8")

    def test_rejects_invalid_ratio_above_twenty_percent_and_preserves_file(self):
        result, contents = self.run_calibration(invalid_every_n=4)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(contents, "preserve-existing-calibration")

    def test_reports_per_imu_counts_when_sample_requirement_is_not_met(self):
        result, contents = self.run_calibration(invalid_every_n=1, min_complete_samples=100)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("per-IMU received/valid/invalid", result.stderr)
        self.assertRegex(result.stderr, r"4=\d+/\d+/\d+")
        self.assertEqual(contents, "preserve-existing-calibration")

    def test_rejects_unknown_imu_id_and_preserves_file(self):
        result, contents = self.run_calibration(invalid_every_n=0, unknown_imu_id=5)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(contents, "preserve-existing-calibration")

    def test_accepts_invalid_ratio_at_twenty_percent(self):
        result, contents = self.run_calibration(invalid_every_n=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calibration = json.loads(contents)
        self.assertGreaterEqual(calibration["sample_count"], 1)
        self.assertEqual(calibration["max_invalid_fraction"], 0.2)
        for name in ("q_01_zero", "q_12_zero", "q_23_zero"):
            norm = sum(component * component for component in calibration[name]) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
