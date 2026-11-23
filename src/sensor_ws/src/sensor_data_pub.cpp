#include "protocol_decoder.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#if defined(__has_include)
#  if __has_include(<sensor_ws/msg/emg_raw.hpp>) && __has_include(<sensor_ws/msg/imu_features.hpp>)
#    include <sensor_ws/msg/emg_raw.hpp>
#    include <sensor_ws/msg/imu_features.hpp>
#    define SENSOR_WS_HAS_CUSTOM_MSGS 1
#  endif
#endif

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace {

speed_t baudConstant(unsigned long baud)
{
  switch (baud) {
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
#ifdef B500000
    case 500000: return B500000;
#endif
    case 921600: return B921600;
#ifdef B1000000
    case 1000000: return B1000000;
#endif
#ifdef B2000000
    case 2000000: return B2000000;
#endif
#ifdef B3000000
    case 3000000: return B3000000;
#endif
    default: throw std::runtime_error("unsupported baud rate on this Linux termios");
  }
}

int openSerial(const std::string &path, unsigned long baud)
{
  const int fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    throw std::runtime_error("open " + path + ": " + std::strerror(errno));
  }
  if (::ioctl(fd, TIOCEXCL) != 0) {
    const std::string error = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("exclusive open " + path + ": " + error);
  }

  termios tty{};
  if (tcgetattr(fd, &tty) != 0) {
    const std::string error = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("tcgetattr: " + error);
  }
  cfmakeraw(&tty);
  const speed_t speed = baudConstant(baud);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;
  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    const std::string error = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("tcsetattr: " + error);
  }
  tcflush(fd, TCIOFLUSH);
  return fd;
}

class SensorDataPublisher final : public rclcpp::Node {
public:
  SensorDataPublisher()
  : Node("sensor_data_pub")
  {
    const std::string port = declare_parameter<std::string>("port", "/dev/ttyACM0");
    const auto baud = declare_parameter<int64_t>("baudrate", 1000000);
    fd_ = openSerial(port, static_cast<unsigned long>(baud));

#ifdef SENSOR_WS_HAS_CUSTOM_MSGS
    emg_pub_ = create_publisher<sensor_ws::msg::EmgRaw>("/emg/raw", 100);
    imu_pub_ = create_publisher<sensor_ws::msg::ImuFeatures>("/imu/raw", 100);
    RCLCPP_INFO(get_logger(), "publishing custom messages on /emg/raw and /imu/raw");
#else
    emg_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/emg/raw", 100);
    imu_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/imu/raw", 100);
    RCLCPP_WARN(
      get_logger(),
      "custom sensor_ws messages are not installed; publishing Float64MultiArray fallback");
#endif

    timer_ = create_wall_timer(std::chrono::milliseconds(1), [this]() { readSerial(); });
    RCLCPP_INFO(get_logger(), "opened %s @ %ld baud", port.c_str(), static_cast<long>(baud));
  }

  ~SensorDataPublisher() override
  {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

private:
  void readSerial()
  {
    std::uint8_t data[8192];
    while (true) {
      const ssize_t count = ::read(fd_, data, sizeof(data));
      if (count > 0) {
        decoder_.feed(data, static_cast<std::size_t>(count),
                      [this](const sensor_protocol::Packet &packet) { publish(packet); });
        continue;
      }
      if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000, "serial read failed: %s", std::strerror(errno));
      }
      break;
    }

    const auto &s = decoder_.stats();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "frames=%lu emg=%lu imu=%lu crc_errors=%lu malformed=%lu startup_drops=%lu",
      static_cast<unsigned long>(s.frames), static_cast<unsigned long>(s.emg_frames),
      static_cast<unsigned long>(s.imu_frames), static_cast<unsigned long>(s.crc_errors),
      static_cast<unsigned long>(s.malformed_frames),
      static_cast<unsigned long>(s.startup_drops));
  }

  void publish(const sensor_protocol::Packet &packet)
  {
    if (packet.kind == sensor_protocol::Packet::Kind::Emg) {
      publishEmg(packet.emg);
    } else {
      publishImu(packet.imu);
    }
  }

  void publishEmg(const sensor_protocol::EmgPacket &p)
  {
#ifdef SENSOR_WS_HAS_CUSTOM_MSGS
    sensor_ws::msg::EmgRaw message;
    message.emg_id = p.emg_id;
    message.emg_seq = p.emg_seq;
    message.sample_index = p.sample_index;
    message.mega_time_us = p.mega_time_us;
    message.grid_time_us = p.grid_time_us;
    for (int i = 0; i < 4; ++i) {
      message.emg_raw[i] = p.emg_raw[i];
    }
    message.valid = p.valid;
    message.header.stamp = now();
    message.header.frame_id = "emg_0";
    emg_pub_->publish(message);
#else
    std_msgs::msg::Float64MultiArray message;
    // [id, seq, sample_index, mega_time_us, grid_time_us, raw0..raw3, valid]
    message.layout.dim.resize(1);
    message.layout.dim[0].label =
      "emg_id emg_seq sample_index mega_time_us grid_time_us emg_raw[4] valid";
    message.layout.dim[0].size = 9;
    message.layout.dim[0].stride = 9;
    message.data = {
      static_cast<double>(p.emg_id), static_cast<double>(p.emg_seq),
      static_cast<double>(p.sample_index), static_cast<double>(p.mega_time_us),
      static_cast<double>(p.grid_time_us), static_cast<double>(p.emg_raw[0]),
      static_cast<double>(p.emg_raw[1]), static_cast<double>(p.emg_raw[2]),
      static_cast<double>(p.emg_raw[3]), p.valid ? 1.0 : 0.0};
    emg_pub_->publish(message);
#endif
  }

  void publishImu(const sensor_protocol::ImuPacket &p)
  {
#ifdef SENSOR_WS_HAS_CUSTOM_MSGS
    sensor_ws::msg::ImuFeatures message;
    message.imu_id = p.imu_id;
    message.imu_seq = p.imu_seq;
    message.sample_index = p.sample_index;
    message.mega_time_us = p.mega_time_us;
    message.grid_time_us = p.grid_time_us;
    for (int i = 0; i < 3; ++i) {
      message.acceleration_g[i] = p.acceleration_g[i];
      message.gyro_dps[i] = p.gyro_dps[i];
      message.angle_deg[i] = p.angle_deg[i];
    }
    for (int i = 0; i < 4; ++i) {
      message.quaternion[i] = p.quaternion[i];
    }
    message.valid = p.valid;
    message.header.stamp = now();
    message.header.frame_id = "imu_" + std::to_string(p.imu_id);
    imu_pub_->publish(message);
#else
    std_msgs::msg::Float64MultiArray message;
    // [id, seq, sample_index, mega_time_us, grid_time_us,
    //  acc[3], gyro[3], angle[3], quaternion[w,x,y,z], valid]
    message.layout.dim.resize(1);
    message.layout.dim[0].label =
      "imu_id imu_seq sample_index mega_time_us grid_time_us acc[3] gyro[3] angle[3] quaternion[4] valid";
    message.layout.dim[0].size = 22;
    message.layout.dim[0].stride = 22;
    message.data.reserve(22);
    message.data.push_back(static_cast<double>(p.imu_id));
    message.data.push_back(static_cast<double>(p.imu_seq));
    message.data.push_back(static_cast<double>(p.sample_index));
    message.data.push_back(static_cast<double>(p.mega_time_us));
    message.data.push_back(static_cast<double>(p.grid_time_us));
    for (int i = 0; i < 3; ++i) message.data.push_back(p.acceleration_g[i]);
    for (int i = 0; i < 3; ++i) message.data.push_back(p.gyro_dps[i]);
    for (int i = 0; i < 3; ++i) message.data.push_back(p.angle_deg[i]);
    for (int i = 0; i < 4; ++i) message.data.push_back(p.quaternion[i]);
    message.data.push_back(p.valid ? 1.0 : 0.0);
    imu_pub_->publish(message);
#endif
  }

  int fd_ = -1;
  sensor_protocol::Decoder decoder_;
  rclcpp::TimerBase::SharedPtr timer_;

#ifdef SENSOR_WS_HAS_CUSTOM_MSGS
  rclcpp::Publisher<sensor_ws::msg::EmgRaw>::SharedPtr emg_pub_;
  rclcpp::Publisher<sensor_ws::msg::ImuFeatures>::SharedPtr imu_pub_;
#else
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr emg_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr imu_pub_;
#endif
};

}  // namespace

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SensorDataPublisher>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "sensor_data_pub: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
}
