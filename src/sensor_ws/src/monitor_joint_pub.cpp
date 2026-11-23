#include "openarm_joint_mapping.hpp"

#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_ws/msg/imu_features.hpp>
#include <sensor_ws/msg/monitor_imu_joint.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

using Json = nlohmann::json;
using sensor_ws::Quaternion;
using MonitorMessage = sensor_ws::msg::MonitorImuJoint;
using ImuMessage = sensor_ws::msg::ImuFeatures;

class MonitorJointPublisher final : public rclcpp::Node {
 public:
  MonitorJointPublisher() : Node("monitor_joint_pub") {
    const auto input_topic = declare_parameter<std::string>("input_topic", "/imu/raw");
    const auto calibration_file = declare_parameter<std::string>(
        "calibration_file", "/home/lylaff/lylaff/src/sensor_ws/config/imu_zero_q.json");
    loadCalibration(calibration_file);
    publisher_ = create_publisher<MonitorMessage>("/imu/joint", 10);
    subscription_ = create_subscription<ImuMessage>(
        input_topic, rclcpp::SensorDataQoS(),
        std::bind(&MonitorJointPublisher::onImu, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "publishing OpenArm-mapped joint references on /imu/joint");
  }

 private:
  struct PendingSample {
    std::array<bool, 4> present{};
    std::array<bool, 4> valid{};
    std::array<Quaternion, 4> orientations{};
    std_msgs::msg::Header latest_header;
  };

  void loadCalibration(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open calibration file: " + path);
    const Json calibration = Json::parse(input);
    if (calibration.at("schema_version") != 1 ||
        calibration.at("quaternion_order") != "wxyz") {
      throw std::runtime_error("unsupported calibration schema or quaternion order");
    }
    const auto& mapping = calibration.at("imu_id_to_segment");
    std::array<bool, 4> seen{};
    for (std::size_t imu = 0; imu < segment_for_imu_.size(); ++imu) {
      const int64_t segment = mapping.at(std::to_string(imu + 1)).get<int64_t>();
      if (segment < 0 || segment >= 4 || seen[static_cast<std::size_t>(segment)]) {
        throw std::runtime_error("calibration IMU mapping must be a permutation of [0,1,2,3]");
      }
      segment_for_imu_[imu] = static_cast<std::size_t>(segment);
      seen[static_cast<std::size_t>(segment)] = true;
    }
    const std::array<std::string, 3> fields{
        "q_01_zero", "q_12_zero", "q_23_zero"};
    for (std::size_t index = 0; index < fields.size(); ++index) {
      const auto values = calibration.at(fields[index]).get<std::array<double, 4>>();
      zeros_[index] = sensor_ws::normalized({values[0], values[1], values[2], values[3]});
    }
  }

  void onImu(const ImuMessage::SharedPtr message) {
    if (message->imu_id < 1 || message->imu_id > 4) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ignoring invalid imu_id");
      return;
    }
    auto& sample = pending_[message->sample_index];
    const std::size_t segment = segment_for_imu_[message->imu_id - 1];
    if (sample.present[segment]) return;
    sample.present[segment] = true;
    sample.valid[segment] = message->valid;
    if (message->valid) {
      try {
        sample.orientations[segment] = sensor_ws::normalized(
            {message->quaternion[0], message->quaternion[1],
             message->quaternion[2], message->quaternion[3]});
      } catch (const std::invalid_argument&) {
        sample.valid[segment] = false;
      }
    }
    if (message->header.stamp.sec > sample.latest_header.stamp.sec ||
        (message->header.stamp.sec == sample.latest_header.stamp.sec &&
         message->header.stamp.nanosec >= sample.latest_header.stamp.nanosec)) {
      sample.latest_header = message->header;
    }
    if (std::all_of(sample.present.begin(), sample.present.end(), [](bool value) { return value; })) {
      publishSample(sample);
      pending_.erase(message->sample_index);
    }
    while (pending_.size() > 128) pending_.erase(pending_.begin());
  }

  void publishSample(const PendingSample& sample) {
    MonitorMessage output;
    output.header = sample.latest_header;
    output.header.frame_id = "openarm_right_arm_base";
    output.human_joint_ref_valid = std::all_of(
        sample.valid.begin(), sample.valid.end(), [](bool value) { return value; });
    output.human_joint_ref_rad.fill(0.0F);
    if (output.human_joint_ref_valid) {
      const std::array<Quaternion, 3> current{
          sensor_ws::relative(sample.orientations[0], sample.orientations[1]),
          sensor_ws::relative(sample.orientations[1], sample.orientations[2]),
          sensor_ws::relative(sample.orientations[2], sample.orientations[3])};
      std::array<Quaternion, 3> delta{};
      for (std::size_t index = 0; index < current.size(); ++index) {
        const Quaternion zero = sensor_ws::relativeToOpenarmFrame(
            zeros_[index], index, index + 1);
        const Quaternion measured = sensor_ws::relativeToOpenarmFrame(
            current[index], index, index + 1);
        delta[index] = sensor_ws::relative(zero, measured);
      }
      const auto angles = sensor_ws::openarmJointAngles(delta[0], delta[1], delta[2]);
      for (std::size_t joint = 0; joint < angles.size(); ++joint) {
        output.human_joint_ref_rad[joint] = static_cast<float>(angles[joint]);
      }
    }
    publisher_->publish(output);
  }

  std::array<std::size_t, 4> segment_for_imu_{};
  std::array<Quaternion, 3> zeros_{};
  std::map<uint32_t, PendingSample> pending_;
  rclcpp::Publisher<MonitorMessage>::SharedPtr publisher_;
  rclcpp::Subscription<ImuMessage>::SharedPtr subscription_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<MonitorJointPublisher>());
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("monitor_joint_pub"), "Publisher stopped: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
