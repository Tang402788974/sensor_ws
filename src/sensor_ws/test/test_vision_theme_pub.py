#!/usr/bin/env python3
"""Black-box checks for the virtual vision publisher; run in a sourced ROS shell."""

import os
import pathlib
import subprocess
import time
import unittest

import rclpy
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from sensor_ws.msg import VisionTheme


class VisionThemePublisherTest(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node("vision_theme_pub_test")
        self.messages = []
        self.subscription = self.node.create_subscription(
            VisionTheme, "/vision/theme", self.messages.append, 10
        )
        executable = os.environ.get("VISION_THEME_PUB_EXECUTABLE")
        if not executable:
            prefix = subprocess.check_output(
                ["ros2", "pkg", "prefix", "sensor_ws"], text=True
            ).strip()
            executable = str(pathlib.Path(prefix) / "lib/sensor_ws/vision_theme_pub")
        command = [executable]
        command += ["--ros-args", "-p", "publish_rate_hz:=20"]
        self.publisher = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self.client = self.node.create_client(SetParameters, "/vision_theme_pub/set_parameters")
        self.assertTrue(self.client.wait_for_service(timeout_sec=5.0))

    def tearDown(self):
        self.publisher.terminate()
        try:
            self.publisher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.publisher.kill()
            self.publisher.wait(timeout=5)
        self.node.destroy_node()
        rclpy.shutdown()

    def wait_for_message(self, predicate, timeout_sec=3.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if any(predicate(message) for message in self.messages):
                return True
        return False

    def set_theme(self, theme_id):
        request = SetParameters.Request()
        request.parameters = [Parameter("theme_id", value=theme_id).to_parameter_msg()]
        future = self.client.call_async(request)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(future.done(), "theme parameter service timed out")
        return future.result().results[0]

    def test_theme_can_change_and_scene_is_fixed_to_s01(self):
        self.assertTrue(
            self.wait_for_message(lambda message: message.theme_id == 1 and message.scene_id == 1),
            "publisher did not emit the initial T_01/S_01 label",
        )

        self.assertTrue(self.set_theme(2).successful)
        self.assertTrue(
            self.wait_for_message(lambda message: message.theme_id == 2 and message.scene_id == 1),
            "publisher did not switch to T_02/S_01",
        )

        self.assertTrue(self.set_theme(3).successful)
        self.assertTrue(
            self.wait_for_message(lambda message: message.theme_id == 3 and message.scene_id == 1),
            "publisher did not switch to T_03/S_01",
        )

        self.assertFalse(self.set_theme(4).successful, "unsupported theme IDs must be rejected")


if __name__ == "__main__":
    unittest.main()
