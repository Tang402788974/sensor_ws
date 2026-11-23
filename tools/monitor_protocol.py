"""JSON Lines contract shared by the desktop app and Pi bridge."""

from __future__ import annotations

import json
import math


TOPIC_TYPES = {
    "/imu/raw": "sensor_ws/msg/ImuFeatures",
    "/emg/raw": "sensor_ws/msg/EmgRaw",
    "/vision/theme": "sensor_ws/msg/VisionTheme",
}


def stamp_ns(stamp) -> int:
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if not 0 <= nanosec < 1_000_000_000:
        raise ValueError("header nanosec is outside ROS time range")
    return sec * 1_000_000_000 + nanosec


def _integer(value, field, minimum=0, maximum=0xFFFFFFFF):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} is outside its integer range")
    return value


def _number_list(values, length, field):
    if len(values) != length:
        raise ValueError(f"{field} must contain {length} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain finite values")
    return result


def _valid_bool(value, field="valid"):
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    kind = event.get("type")
    if kind == "hello":
        if event.get("version") != 1 or event.get("topics") != TOPIC_TYPES:
            raise ValueError("unsupported bridge hello event")
    elif kind == "heartbeat":
        uptime = float(event.get("uptime_s"))
        if not math.isfinite(uptime) or uptime < 0:
            raise ValueError("uptime_s must be a finite non-negative number")
    elif kind == "imu":
        _integer(event.get("header_stamp_ns"), "header_stamp_ns", -0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF)
        _integer(event.get("imu_id"), "imu_id", 1, 4)
        _integer(event.get("imu_seq"), "imu_seq")
        _integer(event.get("sample_index"), "sample_index")
        _number_list(event.get("angle_deg", []), 3, "angle_deg")
        _number_list(event.get("acceleration_g", []), 3, "acceleration_g")
        _number_list(event.get("gyro_dps", []), 3, "gyro_dps")
        _number_list(event.get("quaternion", []), 4, "quaternion")
        _valid_bool(event.get("valid"))
    elif kind == "emg":
        _integer(event.get("header_stamp_ns"), "header_stamp_ns", -0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF)
        _integer(event.get("emg_id"), "emg_id", 0, 255)
        _integer(event.get("emg_seq"), "emg_seq")
        _integer(event.get("sample_index"), "sample_index")
        _number_list(event.get("emg_raw", []), 4, "emg_raw")
        _valid_bool(event.get("valid"))
    elif kind == "vision":
        _integer(event.get("header_stamp_ns"), "header_stamp_ns", -0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF)
        _integer(event.get("theme_id"), "theme_id")
        _integer(event.get("scene_id"), "scene_id")
        _valid_bool(event.get("valid"))
    elif kind == "error":
        if not isinstance(event.get("code"), str) or not event["code"]:
            raise ValueError("error code must be a non-empty string")
        if not isinstance(event.get("message"), str):
            raise ValueError("error message must be a string")
    else:
        raise ValueError("unknown event type")
    return event


def _reject_constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def parse_event_line(line: str | bytes) -> dict:
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        event = json.loads(line.strip(), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON event line") from error
    try:
        return _validate_event(event)
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid event fields") from error


class JsonLineBuffer:
    def __init__(self):
        self._buffer = bytearray()
        self.error_count = 0

    def feed(self, data: bytes) -> list[dict]:
        self._buffer.extend(data)
        events = []
        while b"\n" in self._buffer:
            line, _, remaining = self._buffer.partition(b"\n")
            self._buffer = bytearray(remaining)
            try:
                events.append(parse_event_line(line))
            except ValueError:
                self.error_count += 1
        return events


def make_imu_event(message) -> dict:
    return _validate_event({
        "type": "imu",
        "header_stamp_ns": stamp_ns(message.header.stamp),
        "imu_id": int(message.imu_id),
        "imu_seq": int(message.imu_seq),
        "sample_index": int(message.sample_index),
        "angle_deg": _number_list(message.angle_deg, 3, "angle_deg"),
        "acceleration_g": _number_list(message.acceleration_g, 3, "acceleration_g"),
        "gyro_dps": _number_list(message.gyro_dps, 3, "gyro_dps"),
        "quaternion": _number_list(message.quaternion, 4, "quaternion"),
        "valid": bool(message.valid),
    })


def make_emg_event(message) -> dict:
    return _validate_event({
        "type": "emg",
        "header_stamp_ns": stamp_ns(message.header.stamp),
        "emg_id": int(message.emg_id),
        "emg_seq": int(message.emg_seq),
        "sample_index": int(message.sample_index),
        "emg_raw": _number_list(message.emg_raw, 4, "emg_raw"),
        "valid": bool(message.valid),
    })


def make_vision_event(message) -> dict:
    return _validate_event({
        "type": "vision",
        "header_stamp_ns": stamp_ns(message.header.stamp),
        "theme_id": int(message.theme_id),
        "scene_id": int(message.scene_id),
        "valid": bool(message.valid),
    })


def make_hello_event() -> dict:
    return {"type": "hello", "version": 1, "topics": dict(TOPIC_TYPES)}


def make_heartbeat_event(uptime_s: float) -> dict:
    uptime = float(uptime_s)
    if not math.isfinite(uptime) or uptime < 0:
        raise ValueError("uptime_s must be finite and non-negative")
    return {"type": "heartbeat", "uptime_s": uptime}


def make_error_event(code: str, message: str) -> dict:
    if not isinstance(code, str) or not code or not isinstance(message, str):
        raise ValueError("error event requires a code and message")
    return {"type": "error", "code": code, "message": message}
