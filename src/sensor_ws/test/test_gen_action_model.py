#!/usr/bin/env python3
"""Integration test for the Parquet-to-template C++ program; run on the Pi."""

import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


def make_dataset(path, session_id, start_us, angles):
    columns = {
        "session_id": pa.array([session_id] * len(angles), type=pa.string()),
        "trial_id": pa.array([1] * len(angles), type=pa.uint32()),
        "theme_id": pa.array([2] * len(angles), type=pa.uint32()),
        "grid_time_us": pa.array(
            [start_us + index * 10_000 for index in range(len(angles))], type=pa.uint64()
        ),
        "human_joint_ref_rad": pa.array(angles, type=pa.list_(pa.float32(), 7)),
        "human_joint_ref_valid": pa.array([True] * len(angles)),
    }
    for segment in range(4):
        columns[f"imu_valid_{segment}"] = pa.array([True] * len(angles))
    table = pa.table(columns).replace_schema_metadata(
        {
            b"coordinate_frame": b"openarm_v2_right_arm_base",
            b"schema_version": b"3",
            b"joint_axis_order": b"J1=-Y,J2=-X,J3=-Z,J4=-Y,J5=-Z,J6=+Y,J7=+X",
        }
    )
    pq.write_table(table, path)


class ActionModelIntegrationTest(unittest.TestCase):
    def action_model_executable(self):
        executable = os.environ.get("GEN_ACTION_MODEL_EXECUTABLE")
        if executable:
            return executable
        prefix = subprocess.check_output(
            ["ros2", "pkg", "prefix", "sensor_ws"], text=True
        ).strip()
        return str(pathlib.Path(prefix) / "lib/sensor_ws/gen_action_model")

    def test_averages_trials_on_normalized_timeline_and_writes_required_fields(self):
        executable = self.action_model_executable()

        with tempfile.TemporaryDirectory(prefix="action-model-test-") as directory:
            root = pathlib.Path(directory)
            first = root / "session-a.parquet"
            second = root / "session-b.parquet"
            output = root / "action_model.json"
            make_dataset(first, "session-a", 100_000, [[0.0] * 7, [1.0] * 7, [2.0] * 7])
            make_dataset(second, "session-b", 9_000_000, [[2.0] * 7, [3.0] * 7, [4.0] * 7])

            result = subprocess.run(
                [executable, "--input", str(first), str(second), "--output", str(output)],
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(model["schema_version"], 2)
            self.assertFalse(model["ready_for_actuation"])
            self.assertEqual(model["coordinate_frame"], "openarm_v2_right_arm_base")
            self.assertEqual(
                model["joint_mapping"],
                "IMU axes transformed to OpenArm v2 right-arm base; URDF axis order J1-J7",
            )
            self.assertEqual(len(model["templates"]), 1)
            template = model["templates"][0]
            self.assertEqual(template["theme_id"], 2)
            self.assertEqual(template["trial_count"], 2)
            samples = template["samples"]
            self.assertEqual([sample["grid_time_us"] for sample in samples], [0, 10_000, 20_000])
            self.assertTrue(all(sample["theme_id"] == 2 for sample in samples))
            self.assertTrue(all(
                set(sample) == {"grid_time_us", "theme_id", "prosthesis_joint_ref_rad"}
                for sample in samples
            ))
            for sample, expected in zip(samples, (1.0, 2.0, 3.0)):
                for angle in sample["prosthesis_joint_ref_rad"]:
                    self.assertAlmostEqual(angle, expected, places=5)

    def test_rejects_parquet_without_current_coordinate_frame(self):
        executable = self.action_model_executable()
        with tempfile.TemporaryDirectory(prefix="action-model-old-frame-") as directory:
            root = pathlib.Path(directory)
            source = root / "old.parquet"
            output = root / "action_model.json"
            columns = {
                "session_id": pa.array(["old-session"], type=pa.string()),
                "trial_id": pa.array([1], type=pa.uint32()),
                "theme_id": pa.array([2], type=pa.uint32()),
                "grid_time_us": pa.array([100_000], type=pa.uint64()),
                "human_joint_ref_rad": pa.array([[0.0] * 7], type=pa.list_(pa.float32(), 7)),
                "human_joint_ref_valid": pa.array([True]),
            }
            for segment in range(4):
                columns[f"imu_valid_{segment}"] = pa.array([True])
            pq.write_table(pa.table(columns), source)

            result = subprocess.run(
                [executable, "--input", str(source), "--output", str(output)],
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coordinate frame or joint-axis mapping is missing or outdated", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
