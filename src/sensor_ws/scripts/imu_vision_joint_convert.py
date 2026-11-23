#!/usr/bin/env python3
"""Convert synchronized IMU and vision ROS bag data into a Parquet dataset."""

import argparse
from bisect import bisect_right
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


IMU_TOPIC = "/imu/raw"
VISION_TOPIC = "/vision/theme"
IMU_TYPE = "sensor_ws/msg/ImuFeatures"
VISION_TYPE = "sensor_ws/msg/VisionTheme"
MAX_PENDING_GROUPS = 256
JOINT_ORDER = "J1,J2,J3,J4,J5,J6,J7"
COORDINATE_FRAME = "openarm_v2_right_arm_base"
IMU0_TO_OPENARM = [0.5, -0.5, 0.5, -0.5]
ARM_IMU_TO_OPENARM = [0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5)]
QUATERNION_NAMES = (
    "q_g0_wxyz",
    "q_g1_wxyz",
    "q_g2_wxyz",
    "q_g3_wxyz",
    "q_01",
    "q_12",
    "q_23",
    "q_01_delta",
    "q_12_delta",
    "q_23_delta",
)


def _normalize_quaternion(q):
    values = [float(component) for component in q]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain four finite values")
    norm = math.hypot(math.hypot(values[0], values[1]), math.hypot(values[2], values[3]))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is too small")
    return [value / norm for value in values]


def quaternion_multiply(a, b):
    aw, ax, ay, az = (float(value) for value in a)
    bw, bx, by, bz = (float(value) for value in b)
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quaternion_inverse(q):
    values = [float(component) for component in q]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain four finite values")
    norm_squared = sum(value * value for value in values)
    if norm_squared <= 1e-24:
        raise ValueError("quaternion norm is too small")
    return [values[0] / norm_squared, -values[1] / norm_squared,
            -values[2] / norm_squared, -values[3] / norm_squared]


def quaternion_to_euler_zyx(q):
    w, x, y, z = _normalize_quaternion(q)
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = math.asin(sin_pitch)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return yaw, pitch, roll


def relative_to_openarm_frame(q, proximal_segment, distal_segment):
    proximal_mount = IMU0_TO_OPENARM if proximal_segment == 0 else ARM_IMU_TO_OPENARM
    distal_mount = IMU0_TO_OPENARM if distal_segment == 0 else ARM_IMU_TO_OPENARM
    return _normalize_quaternion(quaternion_multiply(
        quaternion_multiply(proximal_mount, q), quaternion_inverse(distal_mount)
    ))


def _rotation_matrix(q):
    w, x, y, z = _normalize_quaternion(q)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def openarm_joint_angles_from_deltas(q01, q12, q23):
    matrices = [_rotation_matrix(q01), _rotation_matrix(q12), _rotation_matrix(q23)]
    shoulder, elbow, wrist = matrices
    shoulder_first = math.atan2(shoulder[0][2], shoulder[2][2])
    shoulder_second = math.asin(max(-1.0, min(1.0, -shoulder[1][2])))
    shoulder_third = math.atan2(shoulder[1][0], shoulder[1][1])

    elbow_pitch = math.atan2(elbow[0][2], elbow[2][2])
    elbow_yaw = math.atan2(elbow[1][0], elbow[1][1])

    wrist_pitch = math.atan2(-wrist[2][0], wrist[0][0])
    wrist_roll = math.atan2(-wrist[1][2], wrist[1][1])
    return [
        -shoulder_first,
        -shoulder_second,
        -shoulder_third,
        -elbow_pitch,
        -elbow_yaw,
        wrist_pitch,
        wrist_roll,
    ]


def _empty_label():
    return {
        "theme_id": 0,
        "scene_id": 0,
        "label_valid": False,
        "scene_weights": [],
        "theme_weights": [],
        "theme_stamp_ns": None,
    }


def _align_sorted_vision_label(events, stamps, imu_stamp_ns, max_age_ns):
    index = bisect_right(stamps, int(imu_stamp_ns)) - 1
    if index < 0:
        return _empty_label()
    event = events[index]
    age_ns = int(imu_stamp_ns) - int(event["stamp_ns"])
    if not event["valid"] or int(event["theme_id"]) == 0 or age_ns > max_age_ns:
        return {
            "theme_id": 0,
            "scene_id": 0,
            "label_valid": False,
            "scene_weights": [],
            "theme_weights": [],
            "theme_stamp_ns": int(event["stamp_ns"]),
        }
    return {
        "theme_id": int(event["theme_id"]),
        "scene_id": int(event["scene_id"]),
        "label_valid": True,
        "scene_weights": [float(value) for value in event.get("scene_weights", [])],
        "theme_weights": [float(value) for value in event.get("theme_weights", [])],
        "theme_stamp_ns": int(event["stamp_ns"]),
    }


def align_vision_label(events, imu_stamp_ns, max_age_ns):
    if max_age_ns < 0:
        raise ValueError("max_age_ns must be non-negative")
    ordered = sorted(events, key=lambda event: int(event["stamp_ns"]))
    stamps = [int(event["stamp_ns"]) for event in ordered]
    return _align_sorted_vision_label(ordered, stamps, imu_stamp_ns, max_age_ns)


def _validate_calibration(calibration):
    if calibration.get("schema_version") != 1 or calibration.get("quaternion_order") != "wxyz":
        raise ValueError("unsupported calibration schema or quaternion order")
    mapping = calibration.get("imu_id_to_segment", {})
    try:
        segment_by_id = {imu_id: int(mapping[str(imu_id)]) for imu_id in range(1, 5)}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("calibration must map wire IDs 1..4 to segments 0..3") from error
    if sorted(segment_by_id.values()) != [0, 1, 2, 3]:
        raise ValueError("calibration must map wire IDs 1..4 to unique segments 0..3")
    zeros = {}
    for name in ("q_01_zero", "q_12_zero", "q_23_zero"):
        try:
            zeros[name] = _normalize_quaternion(calibration[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"calibration field {name} is not a valid quaternion") from error
    calibration_id = calibration.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError("calibration_id must be a non-empty string")
    return segment_by_id, zeros, calibration_id


def _valid_sample_index(value):
    sample_index = int(value)
    if sample_index < 0 or sample_index > 0xFFFFFFFF:
        raise ValueError("sample_index is outside uint32 range")
    return sample_index


def _header_from_record(record):
    seconds, nanoseconds = divmod(int(record["header_stamp_ns"]), 1_000_000_000)
    if seconds > 0x7FFFFFFF:
        raise ValueError("header timestamp seconds exceed ROS builtin_interfaces/Time range")
    return {
        "stamp": {"sec": seconds, "nanosec": nanoseconds},
        "frame_id": str(record.get("frame_id", "")),
    }


def _build_row(records_by_id, sample_index, vision_events, vision_stamps,
               segment_by_id, zero_quaternions, calibration_id, session_id, max_age_ns):
    segment_quaternions = [None, None, None, None]
    imu_valid = [False, False, False, False]
    for imu_id, record in records_by_id.items():
        if not record["valid"]:
            continue
        try:
            q = _normalize_quaternion(record["quaternion"])
        except (TypeError, ValueError):
            continue
        segment = segment_by_id[imu_id]
        imu_valid[segment] = True
        segment_quaternions[segment] = q

    time_record = max(records_by_id.values(), key=lambda record: record["header_stamp_ns"])
    header_stamp_ns = max(int(record["header_stamp_ns"]) for record in records_by_id.values())
    label = _align_sorted_vision_label(
        vision_events, vision_stamps, header_stamp_ns, max_age_ns
    )
    row = {
        "header": _header_from_record(time_record),
        "session_id": session_id,
        "calibration_id": calibration_id,
        "sample_index": sample_index,
        "grid_time_us": int(time_record["grid_time_us"]),
        "mega_time_us": int(time_record["mega_time_us"]),
        "theme_id": int(label["theme_id"]),
        "trial_id": 0,
        "scene_id": int(label["scene_id"]),
        "scene_weights": list(label["scene_weights"]),
        "theme_weights": list(label["theme_weights"]),
        "human_joint_ref_rad": None,
        "human_joint_ref_valid": False,
    }
    for segment in range(4):
        row[f"imu_valid_{segment}"] = imu_valid[segment]
    for segment, name in enumerate(QUATERNION_NAMES[:4]):
        row[name] = segment_quaternions[segment]
    for name in QUATERNION_NAMES[4:]:
        row[name] = None

    if not all(imu_valid):
        return row

    q_g0, q_g1, q_g2, q_g3 = segment_quaternions
    q_01_sensor = _normalize_quaternion(quaternion_multiply(quaternion_inverse(q_g0), q_g1))
    q_12_sensor = _normalize_quaternion(quaternion_multiply(quaternion_inverse(q_g1), q_g2))
    q_23_sensor = _normalize_quaternion(quaternion_multiply(quaternion_inverse(q_g2), q_g3))
    q_01 = relative_to_openarm_frame(q_01_sensor, 0, 1)
    q_12 = relative_to_openarm_frame(q_12_sensor, 1, 2)
    q_23 = relative_to_openarm_frame(q_23_sensor, 2, 3)
    zero_01 = relative_to_openarm_frame(zero_quaternions["q_01_zero"], 0, 1)
    zero_12 = relative_to_openarm_frame(zero_quaternions["q_12_zero"], 1, 2)
    zero_23 = relative_to_openarm_frame(zero_quaternions["q_23_zero"], 2, 3)
    q_01_delta = _normalize_quaternion(
        quaternion_multiply(quaternion_inverse(zero_01), q_01)
    )
    q_12_delta = _normalize_quaternion(
        quaternion_multiply(quaternion_inverse(zero_12), q_12)
    )
    q_23_delta = _normalize_quaternion(
        quaternion_multiply(quaternion_inverse(zero_23), q_23)
    )
    joint_angles = openarm_joint_angles_from_deltas(q_01_delta, q_12_delta, q_23_delta)

    row.update({
        "q_01": q_01,
        "q_12": q_12,
        "q_23": q_23,
        "q_01_delta": q_01_delta,
        "q_12_delta": q_12_delta,
        "q_23_delta": q_23_delta,
        "human_joint_ref_rad": joint_angles,
        "human_joint_ref_valid": True,
    })
    return row


def _prepare_record(record):
    imu_id = int(record["imu_id"])
    if imu_id < 1 or imu_id > 4:
        raise ValueError(f"unknown imu_id {imu_id}")
    sample_index = _valid_sample_index(record["sample_index"])
    prepared = dict(record)
    prepared["imu_id"] = imu_id
    prepared["sample_index"] = sample_index
    prepared["valid"] = bool(record.get("valid", False))
    prepared["frame_id"] = str(record.get("frame_id", ""))
    for field in ("grid_time_us", "mega_time_us", "header_stamp_ns", "bag_record_time_ns"):
        prepared[field] = int(record.get(field, 0))
        if prepared[field] < 0:
            raise ValueError(f"{field} must be non-negative")
    return prepared


def build_rows(imu_records, vision_events, calibration, session_id, max_age_ns):
    """Build sorted rows from record dictionaries; used by tests and small in-memory jobs."""
    if not session_id:
        raise ValueError("session_id must be non-empty")
    if max_age_ns < 0:
        raise ValueError("max_age_ns must be non-negative")
    segment_by_id, zeros, calibration_id = _validate_calibration(calibration)
    events = sorted(vision_events, key=lambda event: int(event["stamp_ns"]))
    stamps = [int(event["stamp_ns"]) for event in events]
    groups = {}
    for source in imu_records:
        record = _prepare_record(source)
        groups.setdefault(record["sample_index"], {})[record["imu_id"]] = record
    rows = [
        _build_row(groups[index], index, events, stamps, segment_by_id, zeros,
                    calibration_id, session_id, max_age_ns)
        for index in sorted(groups)
    ]
    return list(_assign_trial_ids(rows))


def _assign_trial_ids(rows):
    next_trial_id = 0
    active_theme_id = None
    for row in rows:
        theme_id = int(row["theme_id"])
        if theme_id == 0:
            row["trial_id"] = 0
            active_theme_id = None
            yield row
            continue
        if theme_id != active_theme_id:
            next_trial_id += 1
            if next_trial_id > 0xFFFFFFFF:
                raise ValueError("trial_id exceeds uint32 range")
            active_theme_id = theme_id
        row["trial_id"] = next_trial_id
        yield row


def _iter_rows(imu_records, vision_events, calibration, session_id, max_age_ns):
    """Stream rows while bounding incomplete sample groups to avoid loading a whole bag."""
    if not session_id:
        raise ValueError("session_id must be non-empty")
    if max_age_ns < 0:
        raise ValueError("max_age_ns must be non-negative")
    segment_by_id, zeros, calibration_id = _validate_calibration(calibration)
    events = sorted(vision_events, key=lambda event: int(event["stamp_ns"]))
    stamps = [int(event["stamp_ns"]) for event in events]
    pending = OrderedDict()
    last_emitted = None

    for source in imu_records:
        record = _prepare_record(source)
        index = record["sample_index"]
        if index not in pending and last_emitted is not None:
            delta = (index - last_emitted) & 0xFFFFFFFF
            if delta == 0 or delta >= 0x80000000:
                continue
        pending.setdefault(index, {})[record["imu_id"]] = record
        while pending:
            first_index = next(iter(pending))
            first_records = pending[first_index]
            if len(first_records) < 4 and len(pending) <= MAX_PENDING_GROUPS:
                break
            del pending[first_index]
            yield _build_row(first_records, first_index, events, stamps, segment_by_id,
                             zeros, calibration_id, session_id, max_age_ns)
            last_emitted = first_index

    while pending:
        index, records = pending.popitem(last=False)
        yield _build_row(records, index, events, stamps, segment_by_id, zeros,
                         calibration_id, session_id, max_age_ns)


def iter_rows(imu_records, vision_events, calibration, session_id, max_age_ns):
    yield from _assign_trial_ids(
        _iter_rows(imu_records, vision_events, calibration, session_id, max_age_ns)
    )


_SCHEMA_METADATA = {
    b"schema_version": b"3",
    b"quaternion_order": b"wxyz",
    b"angle_unit": b"rad",
    b"joint_order": JOINT_ORDER.encode("ascii"),
    b"coordinate_frame": COORDINATE_FRAME.encode("ascii"),
    b"imu_axis_mapping": b"imu0:x=right,y=down,z=front;imu1-3:x=back,y=down,z=right",
    b"joint_axis_order": b"J1=-Y,J2=-X,J3=-Z,J4=-Y,J5=-Z,J6=+Y,J7=+X",
    b"header_stamp_semantics": b"latest Pi ROS receive timestamp among the IMUs in a sample_index group",
    b"trial_id_semantics": b"one-based contiguous valid theme segment within one session; zero means no valid theme",
    b"invalid_theme_semantics": b"theme_id=0 and trial_id=0 when the latest preceding vision label is invalid or expired",
    b"vision_stamp_semantics": b"Pi ROS system time at image capture; latest preceding valid event within label age",
    b"weight_fields_semantics": b"empty when the source VisionTheme message does not provide weight arrays",
    b"invalid_fixed_list_semantics": b"Invalid fixed-size vectors contain null elements; use the corresponding validity field",
}


def _fixed_float_list(size):
    return pa.list_(pa.float32(), size)


def _schemas_match(actual, expected):
    if actual.names != expected.names or actual.metadata != expected.metadata:
        return False
    for actual_field, expected_field in zip(actual, expected):
        if actual_field.nullable != expected_field.nullable:
            return False
        if pa.types.is_fixed_size_list(expected_field.type):
            if (not pa.types.is_fixed_size_list(actual_field.type) or
                    actual_field.type.list_size != expected_field.type.list_size or
                    actual_field.type.value_type != expected_field.type.value_type):
                return False
        elif actual_field.type != expected_field.type:
            return False
    return True


SCHEMA = pa.schema([
    pa.field("header", pa.struct([
        pa.field("stamp", pa.struct([
            pa.field("sec", pa.int32(), nullable=False),
            pa.field("nanosec", pa.uint32(), nullable=False),
        ]), nullable=False),
        pa.field("frame_id", pa.string(), nullable=False),
    ]), nullable=False),
    pa.field("session_id", pa.string(), nullable=False),
    pa.field("calibration_id", pa.string(), nullable=False),
    pa.field("sample_index", pa.uint32(), nullable=False),
    pa.field("grid_time_us", pa.uint64(), nullable=False),
    pa.field("mega_time_us", pa.uint64(), nullable=False),
    pa.field("theme_id", pa.uint32(), nullable=False),
    pa.field("trial_id", pa.uint32(), nullable=False),
    pa.field("human_joint_ref_rad", _fixed_float_list(7), nullable=False),
    pa.field("human_joint_ref_valid", pa.bool_(), nullable=False),
    *[pa.field(name, _fixed_float_list(4), nullable=False) for name in QUATERNION_NAMES],
    *[pa.field(f"imu_valid_{segment}", pa.bool_(), nullable=False) for segment in range(4)],
    pa.field("scene_id", pa.uint32(), nullable=False),
    pa.field("scene_weights", pa.list_(pa.float32()), nullable=False),
    pa.field("theme_weights", pa.list_(pa.float32()), nullable=False),
], metadata=_SCHEMA_METADATA)


def _row_for_parquet(row):
    result = dict(row)
    for name in QUATERNION_NAMES:
        if result[name] is None:
            result[name] = [None] * 4
    if result["human_joint_ref_rad"] is None:
        result["human_joint_ref_rad"] = [None] * 7
    return result


def _write_row_stream(rows, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".tmp.", suffix=".parquet",
                                     dir=str(output.parent))
    os.close(fd)
    temp = Path(temp_name)
    writer = None
    row_count = 0
    batch = []
    try:
        writer = pq.ParquetWriter(str(temp), SCHEMA, compression="zstd")
        for row in rows:
            batch.append(_row_for_parquet(row))
            row_count += 1
            if len(batch) >= 4096:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        writer.close()
        writer = None
        if row_count == 0 or pq.read_metadata(temp).num_rows != row_count:
            raise ValueError("Parquet row-count validation failed")
        if not _schemas_match(pq.read_schema(temp), SCHEMA):
            raise ValueError("Parquet schema validation failed")
        os.replace(temp, output)
        return row_count
    finally:
        if writer is not None:
            writer.close()
        if temp.exists():
            temp.unlink()


def write_rows(rows, output_path):
    return _write_row_stream(iter(rows), output_path)


def detect_storage_id(bag_path):
    import rosbag2_py

    try:
        metadata = rosbag2_py.MetadataIo().read_metadata(str(bag_path))
    except Exception as error:
        raise ValueError(f"cannot read ROS bag metadata from {bag_path}: {error}") from error
    storage_id = metadata.storage_identifier
    if not storage_id:
        raise ValueError(f"bag metadata has no storage_identifier: {bag_path}")
    return storage_id


def _open_reader(bag_path, storage_id=None):
    import rosbag2_py

    if storage_id is None:
        storage_id = detect_storage_id(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return reader


def _message_class(type_name):
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_name)


def _deserialize(data, type_name):
    from rclpy.serialization import deserialize_message

    return deserialize_message(data, _message_class(type_name))


def _topic_type_map(reader):
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def _header_stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def read_vision_events(bag_path, storage_id=None):
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    if topic_types.get(VISION_TOPIC) != VISION_TYPE:
        raise ValueError(f"bag must contain {VISION_TOPIC} with type {VISION_TYPE}")
    events = []
    while reader.has_next():
        topic, data, _record_time_ns = reader.read_next()
        if topic != VISION_TOPIC:
            continue
        message = _deserialize(data, VISION_TYPE)
        theme_id = int(message.theme_id)
        scene_id = int(message.scene_id)
        if not (0 <= theme_id <= 0xFFFFFFFF and 0 <= scene_id <= 0xFFFFFFFF):
            raise ValueError("vision theme_id and scene_id must be uint32")
        events.append({
            "stamp_ns": _header_stamp_ns(message.header),
            "theme_id": theme_id,
            "scene_id": scene_id,
            "valid": bool(message.valid),
            "scene_weights": [float(value) for value in getattr(message, "scene_weights", [])],
            "theme_weights": [float(value) for value in getattr(message, "theme_weights", [])],
        })
    return sorted(events, key=lambda event: event["stamp_ns"])


def iter_imu_records_from_bag(bag_path, storage_id=None):
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    if topic_types.get(IMU_TOPIC) != IMU_TYPE:
        raise ValueError(f"bag must contain {IMU_TOPIC} with type {IMU_TYPE}")
    while reader.has_next():
        topic, data, record_time_ns = reader.read_next()
        if topic != IMU_TOPIC:
            continue
        message = _deserialize(data, IMU_TYPE)
        yield {
            "imu_id": int(message.imu_id),
            "sample_index": int(message.sample_index),
            "grid_time_us": int(message.grid_time_us),
            "mega_time_us": int(message.mega_time_us),
            "header_stamp_ns": _header_stamp_ns(message.header),
            "bag_record_time_ns": int(record_time_ns),
            "frame_id": str(message.header.frame_id),
            "quaternion": list(message.quaternion),
            "valid": bool(message.valid),
        }


def convert_bag(bag_path, calibration_path, output_path, session_id,
                max_age_ns=200_000_000, storage_id=None):
    with open(calibration_path, "r", encoding="utf-8") as source:
        calibration = json.load(source)
    if storage_id is None:
        storage_id = detect_storage_id(bag_path)
    vision_events = read_vision_events(bag_path, storage_id)
    imu_records = iter_imu_records_from_bag(bag_path, storage_id)
    rows = iter_rows(imu_records, vision_events, calibration, session_id, max_age_ns)
    return _write_row_stream(rows, output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="ROS 2 bag directory")
    parser.add_argument("--calibration", required=True, help="imu_zero_q.json path")
    parser.add_argument("--output", required=True, help="destination Parquet path")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--label-max-age-ms", type=float, default=200.0)
    args = parser.parse_args(argv)
    if args.label_max_age_ms < 0 or not math.isfinite(args.label_max_age_ms):
        parser.error("--label-max-age-ms must be finite and non-negative")
    max_age_ns = int(round(args.label_max_age_ms * 1_000_000))
    try:
        count = convert_bag(args.bag, args.calibration, args.output,
                            args.session_id, max_age_ns)
    except Exception as error:
        print(f"imu_vision_joint_convert: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
