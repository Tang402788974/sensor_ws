#!/usr/bin/env python3
"""Publish four stationary IMUs with shared sample indices for calibration tests."""

import rclpy
from rclpy.node import Node
from sensor_ws.msg import ImuFeatures


class ImuTestPublisher(Node):
    def __init__(self):
        super().__init__("imu_test_publisher")
        self.publisher = self.create_publisher(ImuFeatures, "/imu/raw", 10)
        self.sample_index = 0
        self.invalid_imu_id = self.declare_parameter("invalid_imu_id", 0).value
        self.invalid_every_n = self.declare_parameter("invalid_every_n", 0).value
        self.unknown_imu_id = self.declare_parameter("unknown_imu_id", 0).value
        self.timer = self.create_timer(0.01, self.publish_frame)

    def publish_frame(self):
        for imu_id in range(1, 5):
            message = ImuFeatures()
            message.imu_id = imu_id
            message.imu_seq = self.sample_index
            message.sample_index = self.sample_index
            message.quaternion = [1.0, 0.0, 0.0, 0.0]
            message.gyro_dps = [0.0, 0.0, 0.0]
            message.acceleration_g = [0.0, 0.0, 1.0]
            message.angle_deg = [0.0, 0.0, 0.0]
            message.valid = True
            if (imu_id == self.invalid_imu_id and self.invalid_every_n > 0 and
                    self.sample_index > 0 and
                    self.sample_index % self.invalid_every_n == 0):
                message.valid = False
            self.publisher.publish(message)
        if self.unknown_imu_id:
            message = ImuFeatures()
            message.imu_id = self.unknown_imu_id
            message.sample_index = self.sample_index
            message.quaternion = [1.0, 0.0, 0.0, 0.0]
            message.valid = True
            self.publisher.publish(message)
        self.sample_index = (self.sample_index + 1) & 0xFFFFFFFF


def main():
    rclpy.init()
    node = ImuTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
