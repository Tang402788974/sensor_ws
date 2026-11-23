#include "imu_quaternion.hpp"

#include <sensor_ws/msg/imu_features.hpp>
#include <rclcpp/rclcpp.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sensor_ws::Quaternion;
using Sample = std::array<Quaternion, 4>;
using Json = nlohmann::json;

std::string makeUuid() {
  std::random_device device;
  std::mt19937_64 rng(device());
  std::uniform_int_distribution<unsigned int> byte(0, 255);
  std::array<unsigned int, 16> bytes{};
  for (auto& value : bytes) value = byte(rng);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    out << std::setw(2) << bytes[i];
    if (i == 3 || i == 5 || i == 7 || i == 9) out << '-';
  }
  return out.str();
}

std::string utcNow() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

class ZeroCalibration final : public rclcpp::Node {
 public:
  ZeroCalibration() : Node("imu_calib_zero") {
    const std::string topic = declare_parameter<std::string>("input_topic", "/imu/raw");
    output_path_ = declare_parameter<std::string>(
        "calibration_file", "/home/lylaff/lylaff/src/sensor_ws/config/imu_zero_q.json");
    duration_ = declare_parameter<double>("duration_sec", 3.0);
    min_samples_ = declare_parameter<int>("min_complete_samples", 240);
    max_gyro_rms_dps_ = declare_parameter<double>("max_gyro_rms_dps", 5.0);
    max_invalid_fraction_ = declare_parameter<double>("max_invalid_fraction", 0.2);
    const auto mapping = declare_parameter<std::vector<int64_t>>(
        "imu_id_to_segment", {0, 1, 2, 3});
    if (!(duration_ > 0.0) || min_samples_ <= 0 || !(max_gyro_rms_dps_ >= 0.0) ||
        !(max_invalid_fraction_ >= 0.0) || max_invalid_fraction_ > 1.0 || mapping.size() != 4) {
      throw std::invalid_argument("invalid calibration parameters");
    }
    std::array<bool, 4> seen{};
    for (std::size_t i = 0; i < mapping.size(); ++i) {
      if (mapping[i] < 0 || mapping[i] >= 4 || seen[static_cast<std::size_t>(mapping[i])]) {
        throw std::invalid_argument("imu_id_to_segment must be a permutation of [0,1,2,3]");
      }
      seen[static_cast<std::size_t>(mapping[i])] = true;
      segment_for_id_[i] = static_cast<std::size_t>(mapping[i]);
    }
    duration_timer_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {
      if (started_ && !finished_ &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count() >=
              duration_) {
        finish();
      }
    });
    subscription_ = create_subscription<sensor_ws::msg::ImuFeatures>(
        topic, rclcpp::SensorDataQoS(),
        std::bind(&ZeroCalibration::onMessage, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Collecting stationary IMU samples from %s", topic.c_str());
  }

  int resultCode() const { return result_code_; }

 private:
  void fail(const std::string& reason) {
    if (finished_) return;
    finished_ = true;
    result_code_ = 1;
    RCLCPP_ERROR(get_logger(), "%s; calibration not written", reason.c_str());
    rclcpp::shutdown();
  }

  void recordInvalid(std::size_t id) {
    ++invalid_count_;
    ++invalid_counts_[id];
  }

  void onMessage(const sensor_ws::msg::ImuFeatures::SharedPtr message) {
    if (finished_) return;
    if (message->imu_id < 1 || message->imu_id > 4) {
      fail("unknown imu_id " + std::to_string(message->imu_id));
      return;
    }
    if (!started_) {
      start_ = std::chrono::steady_clock::now();
      started_ = true;
    }
    const std::size_t id = message->imu_id - 1;
    ++received_counts_[id];
    if (!message->valid) {
      recordInvalid(id);
      return;
    }

    Quaternion q{};
    try {
      q = sensor_ws::normalized({message->quaternion[0], message->quaternion[1],
                                 message->quaternion[2], message->quaternion[3]});
    } catch (const std::invalid_argument&) {
      recordInvalid(id);
      return;
    }
    const double gx = message->gyro_dps[0];
    const double gy = message->gyro_dps[1];
    const double gz = message->gyro_dps[2];
    if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) {
      recordInvalid(id);
      return;
    }

    ++valid_counts_[id];
    auto& entry = pending_[message->sample_index];
    if (entry.present[id]) return;
    entry.present[id] = true;
    entry.quaternions[segment_for_id_[id]] = q;
    entry.gyro_sq[id] = gx * gx + gy * gy + gz * gz;
    if (!std::all_of(entry.present.begin(), entry.present.end(), [](bool v) { return v; })) return;
    const Sample sample = entry.quaternions;
    for (std::size_t imu = 0; imu < 4; ++imu) gyro_sq_sum_[imu] += entry.gyro_sq[imu];
    pending_.erase(message->sample_index);
    samples_.push_back(sample);
  }

  void finish() {
    if (static_cast<int>(samples_.size()) < min_samples_) {
      std::ostringstream reason;
      reason << "only " << samples_.size() << " complete valid frames (minimum " << min_samples_
             << "); per-IMU received/valid/invalid: ";
      for (std::size_t i = 0; i < received_counts_.size(); ++i) {
        if (i > 0) reason << ", ";
        reason << i + 1 << "=" << received_counts_[i] << "/" << valid_counts_[i] << "/"
               << invalid_counts_[i];
      }
      reason << "; pending sample indices=" << pending_.size();
      fail(reason.str());
      return;
    }
    std::array<double, 4> gyro_rms{};
    for (std::size_t i = 0; i < 4; ++i) {
      if (received_counts_[i] == 0 ||
          static_cast<double>(invalid_counts_[i]) > max_invalid_fraction_ * received_counts_[i]) {
        const double fraction = received_counts_[i] == 0
                                    ? 1.0
                                    : static_cast<double>(invalid_counts_[i]) / received_counts_[i];
        fail("IMU " + std::to_string(i + 1) + " invalid-message fraction " +
             std::to_string(fraction) + " exceeds " + std::to_string(max_invalid_fraction_));
        return;
      }
      gyro_rms[i] = std::sqrt(gyro_sq_sum_[i] / static_cast<double>(samples_.size()));
      if (!std::isfinite(gyro_rms[i]) || gyro_rms[i] > max_gyro_rms_dps_) {
        fail("IMU " + std::to_string(i + 1) + " gyro RMS " + std::to_string(gyro_rms[i]) +
             " deg/s exceeds " + std::to_string(max_gyro_rms_dps_));
        return;
      }
    }
    std::array<std::vector<Quaternion>, 3> relatives;
    for (const auto& frame : samples_) {
      relatives[0].push_back(sensor_ws::relative(frame[0], frame[1]));
      relatives[1].push_back(sensor_ws::relative(frame[1], frame[2]));
      relatives[2].push_back(sensor_ws::relative(frame[2], frame[3]));
    }
    auto asArray = [](const Quaternion& q) { return Json::array({q.w, q.x, q.y, q.z}); };
    const auto q01 = sensor_ws::signCorrectedMean(relatives[0]);
    const auto q12 = sensor_ws::signCorrectedMean(relatives[1]);
    const auto q23 = sensor_ws::signCorrectedMean(relatives[2]);
    Json mapping = Json::object();
    Json counts = Json::object();
    for (std::size_t i = 0; i < 4; ++i) {
      mapping[std::to_string(i + 1)] = segment_for_id_[i];
      counts[std::to_string(i + 1)] = valid_counts_[i];
    }
    Json result{{"schema_version", 1}, {"calibration_id", makeUuid()},
                {"created_at_utc", utcNow()}, {"quaternion_order", "wxyz"},
                {"zero_pose", "right_arm_down_thumb_forward"},
                {"imu_id_to_segment", mapping}, {"sample_count", samples_.size()},
                {"imu_sample_count", counts}, {"max_invalid_fraction", max_invalid_fraction_},
                {"gyro_rms_limit_dps", max_gyro_rms_dps_},
                {"q_01_zero", asArray(q01)}, {"q_12_zero", asArray(q12)},
                {"q_23_zero", asArray(q23)}};
    std::filesystem::path temp;
    try {
      const std::filesystem::path target(output_path_);
      temp = target.string() + ".tmp." + makeUuid();
      {
        std::ofstream out(temp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open temporary calibration file");
        out << result.dump(2) << '\n';
        out.flush();
        if (!out) throw std::runtime_error("failed writing temporary calibration file");
      }
      const Json check = Json::parse(std::ifstream(temp));
      if (check.at("schema_version") != 1 || check.at("sample_count") != samples_.size()) {
        throw std::runtime_error("temporary calibration validation failed");
      }
      std::filesystem::rename(temp, target);
      finished_ = true;
      RCLCPP_INFO(get_logger(), "Calibration saved to %s (%zu frames, %zu invalid messages)",
                  output_path_.c_str(), samples_.size(), invalid_count_);
    } catch (const std::exception& error) {
      std::error_code ignored;
      if (!temp.empty()) std::filesystem::remove(temp, ignored);
      fail(std::string("calibration write failed: ") + error.what());
    }
    rclcpp::shutdown();
  }

  struct Pending {
    std::array<bool, 4> present{};
    std::array<double, 4> gyro_sq{};
    Sample quaternions{};
  };
  rclcpp::Subscription<sensor_ws::msg::ImuFeatures>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr duration_timer_;
  std::string output_path_;
  double duration_{3.0};
  int min_samples_{240};
  double max_gyro_rms_dps_{5.0};
  double max_invalid_fraction_{0.2};
  std::array<std::size_t, 4> segment_for_id_{};
  std::array<std::size_t, 4> received_counts_{};
  std::array<std::size_t, 4> valid_counts_{};
  std::array<std::size_t, 4> invalid_counts_{};
  std::array<double, 4> gyro_sq_sum_{};
  std::map<std::uint32_t, Pending> pending_;
  std::vector<Sample> samples_;
  std::chrono::steady_clock::time_point start_{};
  std::size_t invalid_count_{0};
  int result_code_{0};
  bool started_{false};
  bool finished_{false};
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    const auto node = std::make_shared<ZeroCalibration>();
    rclcpp::spin(node);
    const int result = node->resultCode();
    rclcpp::shutdown();
    return result;
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("imu_calib_zero"), "Calibration aborted: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
