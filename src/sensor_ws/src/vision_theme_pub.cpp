#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_ws/msg/vision_theme.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

constexpr int64_t kFirstThemeId = 1;
constexpr int64_t kLastThemeId = 3;
constexpr uint32_t kSceneId = 1;

}  // namespace

class VisionThemePublisher final : public rclcpp::Node {
 public:
  VisionThemePublisher() : Node("vision_theme_pub") {
    const auto theme_id = declare_parameter<int64_t>("theme_id", kFirstThemeId);
    const int64_t rate_hz = declare_parameter<int64_t>("publish_rate_hz", 10);
    const bool valid = declare_parameter<bool>("valid", true);
    if (theme_id < kFirstThemeId || theme_id > kLastThemeId) {
      throw std::invalid_argument("theme_id must be 1 (T_01), 2 (T_02), or 3 (T_03)");
    }
    if (rate_hz < 1 || rate_hz > 100) {
      throw std::invalid_argument("publish_rate_hz must be between 1 and 100");
    }

    message_.theme_id = static_cast<uint32_t>(theme_id);
    message_.scene_id = kSceneId;
    message_.valid = valid;
    publisher_ = create_publisher<sensor_ws::msg::VisionTheme>("/vision/theme", 10);
    parameter_callback_handle_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          return update_theme(parameters);
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(1000 / rate_hz), [this]() {
      message_.header.stamp = now();
      message_.header.frame_id = "camera";
      publisher_->publish(message_);
    });
    RCLCPP_INFO(get_logger(), "publishing theme=T_%02u scene=S_01 valid=%s on /vision/theme",
                message_.theme_id, message_.valid ? "true" : "false");
  }

 private:
  rcl_interfaces::msg::SetParametersResult update_theme(
      const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    std::optional<uint32_t> next_theme_id;
    for (const auto& parameter : parameters) {
      if (parameter.get_name() != "theme_id") {
        result.successful = false;
        result.reason = "only theme_id can be changed at runtime";
        return result;
      }
      const int64_t value = parameter.as_int();
      if (value < kFirstThemeId || value > kLastThemeId) {
        result.successful = false;
        result.reason = "theme_id must be 1 (T_01), 2 (T_02), or 3 (T_03)";
        return result;
      }
      next_theme_id = static_cast<uint32_t>(value);
    }
    if (next_theme_id) {
      message_.theme_id = *next_theme_id;
      RCLCPP_INFO(get_logger(), "theme changed to T_%02u", message_.theme_id);
    }
    result.successful = true;
    return result;
  }

  sensor_ws::msg::VisionTheme message_;
  rclcpp::Publisher<sensor_ws::msg::VisionTheme>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<VisionThemePublisher>());
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("vision_theme_pub"), "Publisher aborted: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
