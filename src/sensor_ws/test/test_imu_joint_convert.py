#!/usr/bin/env python3
"""Unit tests for the offline IMU-to-Parquet converter."""

import math
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from imu_vision_joint_convert import (  # noqa: E402
    align_vision_label,
    build_rows,
    convert_bag,
    detect_storage_id,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_zyx,
    openarm_joint_angles_from_deltas,
    relative_to_openarm_frame,
    IMU0_TO_OPENARM,
    ARM_IMU_TO_OPENARM,
    SCHEMA,
    write_rows,
)


def quaternion_from_euler(yaw, pitch, roll):
    half = 0.5
    qz = [math.cos(yaw * half), 0.0, 0.0, math.sin(yaw * half)]
    qy = [math.cos(pitch * half), 0.0, math.sin(pitch * half), 0.0]
    qx = [math.cos(roll * half), math.sin(roll * half), 0.0, 0.0]
    return quaternion_multiply(quaternion_multiply(qz, qy), qx)


def calibration():
    return {
        "schema_version": 1,
        "calibration_id": "cal-test",
        "quaternion_order": "wxyz",
        "imu_id_to_segment": {"1": 0, "2": 1, "3": 2, "4": 3},
        "q_01_zero": quaternion_multiply(
            quaternion_inverse(IMU0_TO_OPENARM), ARM_IMU_TO_OPENARM
        ),
        "q_12_zero": [1.0, 0.0, 0.0, 0.0],
        "q_23_zero": [1.0, 0.0, 0.0, 0.0],
    }


def imu_record(imu_id, quaternion, valid=True, sample_index=10):
    return {
        "imu_id": imu_id,
        "sample_index": sample_index,
        "grid_time_us": 100000,
        "mega_time_us": 100123,
        "header_stamp_ns": 1000000000 + imu_id,
        "bag_record_time_ns": 1000000100 + imu_id,
        "quaternion": quaternion,
        "valid": valid,
    }


class QuaternionTest(unittest.TestCase):
    def test_multiply_inverse_and_zyx(self):
        q = quaternion_from_euler(0.4, 0.3, 0.2)
        identity = quaternion_multiply(q, quaternion_inverse(q))
        self.assertAlmostEqual(identity[0], 1.0, places=12)
        self.assertAlmostEqual(identity[1], 0.0, places=12)
        yaw, pitch, roll = quaternion_to_euler_zyx(q)
        self.assertAlmostEqual(yaw, 0.4, places=12)
        self.assertAlmostEqual(pitch, 0.3, places=12)
        self.assertAlmostEqual(roll, 0.2, places=12)

    def test_maps_sensor_axes_and_joint_signs_to_openarm_v2(self):
        expected = [0.2, -0.25, 0.3, -0.35, 0.4, -0.45, 0.5]

        def axis_rotation(axis, angle):
            values = [0.0, 0.0, 0.0]
            values[axis] = math.sin(angle / 2.0)
            return [math.cos(angle / 2.0), *values]

        def sequence(*rotations):
            result = [1.0, 0.0, 0.0, 0.0]
            for rotation in rotations:
                result = quaternion_multiply(result, rotation)
            return result

        imu0_mount = [0.5, -0.5, 0.5, -0.5]
        arm_mount = [0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5)]

        def robot_to_sensor(q, proximal_mount, distal_mount):
            return quaternion_multiply(
                quaternion_multiply(quaternion_inverse(proximal_mount), q), distal_mount
            )

        q01_robot = sequence(
            axis_rotation(1, -expected[0]),
            axis_rotation(0, -expected[1]),
            axis_rotation(2, -expected[2]),
        )
        q12_robot = sequence(
            axis_rotation(1, -expected[3]),
            axis_rotation(2, -expected[4]),
        )
        q23_robot = sequence(
            axis_rotation(1, expected[5]),
            axis_rotation(0, expected[6]),
        )

        actual = openarm_joint_angles_from_deltas(
            relative_to_openarm_frame(
                robot_to_sensor(q01_robot, imu0_mount, arm_mount), 0, 1
            ),
            relative_to_openarm_frame(
                robot_to_sensor(q12_robot, arm_mount, arm_mount), 1, 2
            ),
            relative_to_openarm_frame(
                robot_to_sensor(q23_robot, arm_mount, arm_mount), 2, 3
            ),
        )

        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target, places=6)


class VisionAlignmentTest(unittest.TestCase):
    def test_latest_invalid_event_resets_previous_valid_label(self):
        events = [
            {"stamp_ns": 100, "theme_id": 2, "scene_id": 3, "valid": True},
            {"stamp_ns": 180, "theme_id": 0, "scene_id": 0, "valid": False},
        ]
        label = align_vision_label(events, imu_stamp_ns=190, max_age_ns=100)
        self.assertFalse(label["label_valid"])
        self.assertEqual(label["theme_id"], 0)
        self.assertEqual(label["theme_stamp_ns"], 180)

    def test_label_expires_after_maximum_age(self):
        events = [{"stamp_ns": 100, "theme_id": 2, "scene_id": 3, "valid": True}]
        label = align_vision_label(events, imu_stamp_ns=201, max_age_ns=100)
        self.assertFalse(label["label_valid"])
        self.assertEqual(label["theme_stamp_ns"], 100)


class RowConversionTest(unittest.TestCase):
    def test_assigns_one_trial_id_per_contiguous_theme_segment(self):
        records = []
        for sample_index, stamp_ns in ((10, 100), (11, 200), (12, 300), (13, 400)):
            for imu_id in range(1, 5):
                record = imu_record(imu_id, [1.0, 0.0, 0.0, 0.0], sample_index=sample_index)
                record["header_stamp_ns"] = stamp_ns + imu_id
                record["frame_id"] = f"imu_{imu_id - 1}"
                records.append(record)
        events = [
            {"stamp_ns": 0, "theme_id": 1, "scene_id": 1, "valid": True},
            {"stamp_ns": 150, "theme_id": 2, "scene_id": 1, "valid": True},
            {"stamp_ns": 350, "theme_id": 3, "scene_id": 1, "valid": True},
        ]

        rows = build_rows(records, events, calibration(), "session-test", 1000)

        self.assertEqual([row["theme_id"] for row in rows], [1, 2, 2, 3])
        self.assertEqual([row["trial_id"] for row in rows], [1, 2, 2, 3])
        self.assertEqual(rows[0]["header"]["frame_id"], "imu_3")
        self.assertEqual(rows[0]["scene_weights"], [])
        self.assertEqual(rows[0]["theme_weights"], [])

    def test_parquet_schema_matches_output_msg_and_has_no_prosthesis_measurements(self):
        expected = [
            "header", "session_id", "calibration_id", "sample_index", "grid_time_us",
            "mega_time_us", "theme_id", "trial_id", "human_joint_ref_rad",
            "human_joint_ref_valid", "q_g0_wxyz", "q_g1_wxyz", "q_g2_wxyz", "q_g3_wxyz",
            "q_01", "q_12", "q_23", "q_01_delta", "q_12_delta", "q_23_delta",
            "imu_valid_0", "imu_valid_1", "imu_valid_2", "imu_valid_3", "scene_id",
            "scene_weights", "theme_weights",
        ]
        self.assertEqual(SCHEMA.names, expected)
        self.assertNotIn("prosthesis_joint_ref_rad", SCHEMA.names)
        self.assertNotIn("prosthesis_joint_ref_valid", SCHEMA.names)

    def test_maps_openarm_axis_sequences_and_round_trips_parquet(self):
        expected = [0.2, -0.25, 0.3, -0.35, 0.4, -0.45, 0.5]

        def axis_rotation(axis, angle):
            values = [0.0, 0.0, 0.0]
            values[axis] = math.sin(angle / 2.0)
            return [math.cos(angle / 2.0), *values]

        def sequence(*rotations):
            result = [1.0, 0.0, 0.0, 0.0]
            for rotation in rotations:
                result = quaternion_multiply(result, rotation)
            return result

        q01_robot = sequence(
            axis_rotation(1, -expected[0]),
            axis_rotation(0, -expected[1]),
            axis_rotation(2, -expected[2]),
        )
        q12_robot = sequence(
            axis_rotation(1, -expected[3]),
            axis_rotation(2, -expected[4]),
        )
        q23_robot = sequence(
            axis_rotation(1, expected[5]),
            axis_rotation(0, expected[6]),
        )
        q01 = quaternion_multiply(
            quaternion_multiply(quaternion_inverse(IMU0_TO_OPENARM), q01_robot),
            ARM_IMU_TO_OPENARM,
        )
        q12 = quaternion_multiply(
            quaternion_multiply(quaternion_inverse(ARM_IMU_TO_OPENARM), q12_robot),
            ARM_IMU_TO_OPENARM,
        )
        q23 = quaternion_multiply(
            quaternion_multiply(quaternion_inverse(ARM_IMU_TO_OPENARM), q23_robot),
            ARM_IMU_TO_OPENARM,
        )
        qg = [
            [1.0, 0.0, 0.0, 0.0],
            q01,
            quaternion_multiply(q01, q12),
            quaternion_multiply(quaternion_multiply(q01, q12), q23),
        ]
        records = [imu_record(i + 1, qg[i]) for i in range(4)]
        events = [{"stamp_ns": 999999900, "theme_id": 7, "scene_id": 8, "valid": True}]

        rows = build_rows(records, events, calibration(), "session-test", 200)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for actual, target in zip(row["human_joint_ref_rad"], expected):
            self.assertAlmostEqual(actual, target, places=6)
        self.assertTrue(row["human_joint_ref_valid"])
        self.assertEqual(row["theme_id"], 7)
        self.assertEqual(row["trial_id"], 1)
        self.assertEqual(row["scene_id"], 8)
        self.assertEqual(row["scene_weights"], [])
        self.assertEqual(row["theme_weights"], [])

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "dataset.parquet"
            write_rows(rows, output)
            table = pq.read_table(output)
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table["sample_index"].type, pa.uint32())
            self.assertEqual(table["theme_id"].type, pa.uint32())
            self.assertEqual(table["trial_id"].type, pa.uint32())
            self.assertEqual(table["grid_time_us"].type, pa.uint64())
            self.assertEqual(table["human_joint_ref_rad"].type, pa.list_(pa.float32(), 7))
            self.assertEqual(table["q_g0_wxyz"].type, pa.list_(pa.float32(), 4))
            self.assertEqual(table["header"].type.field("frame_id").type, pa.string())
            self.assertEqual(table.schema.metadata[b"angle_unit"], b"rad")
            self.assertEqual(
                table.schema.metadata[b"coordinate_frame"], b"openarm_v2_right_arm_base"
            )
            self.assertEqual(
                table.schema.metadata[b"joint_axis_order"],
                b"J1=-Y,J2=-X,J3=-Z,J4=-Y,J5=-Z,J6=+Y,J7=+X",
            )
            self.assertEqual(table["imu_valid_0"].to_pylist(), [True])
            self.assertNotIn("prosthesis_joint_ref_rad", table.schema.names)
            self.assertNotIn("prosthesis_joint_ref_valid", table.schema.names)

    def test_incomplete_imu_group_has_no_joint_target(self):
        records = [imu_record(i, [1.0, 0.0, 0.0, 0.0]) for i in (1, 2, 3)]
        rows = build_rows(records, [], calibration(), "session-test", 200)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["imu_valid_3"])
        self.assertFalse(rows[0]["human_joint_ref_valid"])
        self.assertIsNone(rows[0]["human_joint_ref_rad"])
        self.assertEqual(rows[0]["theme_id"], 0)
        self.assertEqual(rows[0]["trial_id"], 0)

    def test_empty_conversion_does_not_replace_existing_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "dataset.parquet"
            output.write_bytes(b"keep-existing-file")
            with self.assertRaises(ValueError):
                write_rows([], output)
            self.assertEqual(output.read_bytes(), b"keep-existing-file")


class RosbagConversionTest(unittest.TestCase):
    def test_detects_storage_identifier_from_rosbag_metadata(self):
        import rosbag2_py

        with patch.object(rosbag2_py.MetadataIo, "read_metadata") as read_metadata:
            read_metadata.return_value.storage_identifier = "mcap"
            self.assertEqual(detect_storage_id("/tmp/a-mcap-bag"), "mcap")
            read_metadata.assert_called_once_with("/tmp/a-mcap-bag")

    def test_synthetic_rosbag_converts_complete_and_incomplete_samples(self):
        import rosbag2_py
        from rclpy.serialization import serialize_message
        from sensor_ws.msg import ImuFeatures, VisionTheme

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bag_path = root / "bag"
            calibration_path = root / "imu_zero_q.json"
            output_path = root / "dataset.parquet"
            calibration_path.write_text(json.dumps(calibration()), encoding="utf-8")

            writer = rosbag2_py.SequentialWriter()
            writer.open(
                rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"),
            )
            writer.create_topic(rosbag2_py.TopicMetadata(
                id=0, name="/imu/raw", type="sensor_ws/msg/ImuFeatures", serialization_format="cdr"
            ))
            writer.create_topic(rosbag2_py.TopicMetadata(
                id=1, name="/vision/theme", type="sensor_ws/msg/VisionTheme", serialization_format="cdr"
            ))

            vision = VisionTheme()
            vision.header.stamp.sec = 0
            vision.header.stamp.nanosec = 999_999_900
            vision.theme_id = 7
            vision.scene_id = 8
            vision.valid = True
            writer.write("/vision/theme", serialize_message(vision), 999_999_900)

            for imu_id in range(1, 5):
                imu = ImuFeatures()
                imu.header.stamp.sec = 1
                imu.header.stamp.nanosec = imu_id
                imu.imu_id = imu_id
                imu.sample_index = 10
                imu.grid_time_us = 100_000
                imu.mega_time_us = 100_123
                imu.quaternion = [1.0, 0.0, 0.0, 0.0]
                imu.valid = True
                writer.write("/imu/raw", serialize_message(imu), 1_000_000_000 + imu_id)
            for imu_id in range(1, 4):
                imu = ImuFeatures()
                imu.header.stamp.sec = 1
                imu.header.stamp.nanosec = 10_000 + imu_id
                imu.imu_id = imu_id
                imu.sample_index = 11
                imu.grid_time_us = 110_000
                imu.mega_time_us = 110_123
                imu.quaternion = [1.0, 0.0, 0.0, 0.0]
                imu.valid = True
                writer.write("/imu/raw", serialize_message(imu), 1_000_010_000 + imu_id)
            del writer

            row_count = convert_bag(
                bag_path, calibration_path, output_path, "session-bag-test", 200_000_000
            )
            table = pq.read_table(output_path)
            self.assertEqual(row_count, 2)
            self.assertEqual(table["sample_index"].to_pylist(), [10, 11])
            self.assertEqual(table["theme_id"].to_pylist(), [7, 7])
            self.assertEqual(table["trial_id"].to_pylist(), [1, 1])
            self.assertEqual(table["scene_id"].to_pylist(), [8, 8])
            self.assertEqual(table["human_joint_ref_valid"].to_pylist(), [True, False])
            self.assertEqual(table["imu_valid_3"].to_pylist(), [True, False])
            self.assertEqual(table["human_joint_ref_rad"][1].as_py(), [None] * 7)
            self.assertEqual(table["q_g3_wxyz"][1].as_py(), [None] * 4)
            self.assertEqual(table["header"][0].as_py()["stamp"], {"sec": 1, "nanosec": 4})
            self.assertEqual(table["scene_weights"].to_pylist(), [[], []])
            self.assertEqual(table["theme_weights"].to_pylist(), [[], []])


if __name__ == "__main__":
    unittest.main()
