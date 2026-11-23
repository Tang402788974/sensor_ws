"""Expose the Pi sensor topics as newline-delimited JSON over SSH."""

from __future__ import annotations

import json
import select
import sys
import time

if __package__:
    from .monitor_protocol import (
        TOPIC_TYPES,
        make_emg_event,
        make_error_event,
        make_hello_event,
        make_heartbeat_event,
        make_imu_event,
        make_vision_event,
    )
else:
    from monitor_protocol import (
        TOPIC_TYPES,
        make_emg_event,
        make_error_event,
        make_hello_event,
        make_heartbeat_event,
        make_imu_event,
        make_vision_event,
    )


def topic_manifest() -> dict[str, str]:
    return dict(TOPIC_TYPES)


def emit_event(event: dict, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _emit_message(converter, code):
    def callback(message):
        try:
            emit_event(converter(message))
        except Exception as error:
            emit_event(make_error_event(code, str(error)))
    return callback


def main() -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_ws.msg import EmgRaw, ImuFeatures, VisionTheme
    except Exception as error:
        emit_event(make_error_event("bridge_startup", str(error)))
        return 1

    rclpy.init()
    node = Node("desktop_sensor_monitor_bridge")
    started = time.monotonic()
    subscriptions = [
        node.create_subscription(
            ImuFeatures, "/imu/raw", _emit_message(make_imu_event, "invalid_imu"), 100
        ),
        node.create_subscription(
            EmgRaw, "/emg/raw", _emit_message(make_emg_event, "invalid_emg"), 100
        ),
        node.create_subscription(
            VisionTheme, "/vision/theme", _emit_message(make_vision_event, "invalid_vision"), 100
        ),
    ]
    heartbeat = node.create_timer(
        1.0, lambda: emit_event(make_heartbeat_event(time.monotonic() - started))
    )
    emit_event(make_hello_event())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                command = sys.stdin.readline()
                if not command or command.strip() == "stop":
                    break
    except KeyboardInterrupt:
        pass
    except Exception as error:
        emit_event(make_error_event("bridge_runtime", str(error)))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        del subscriptions, heartbeat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
