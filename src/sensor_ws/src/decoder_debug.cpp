#include "protocol_decoder.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t g_running = 1;

void onSignal(int) { g_running = 0; }

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

struct Options {
  std::string port = "/dev/ttyACM0";
  unsigned long baud = 1000000;
  double seconds = 0.0;
  bool verbose = false;
};

void usage(const char *program)
{
  std::cerr << "Usage: " << program
            << " [--port /dev/ttyACM0] [--baud 1000000]"
            << " [--seconds N] [--verbose]\n";
}

Options parseOptions(int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (argument == "--port") {
      options.port = next("--port");
    } else if (argument == "--baud") {
      options.baud = std::stoul(next("--baud"));
    } else if (argument == "--seconds") {
      options.seconds = std::stod(next("--seconds"));
    } else if (argument == "--verbose") {
      options.verbose = true;
    } else if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

void printPacket(const sensor_protocol::Packet &packet)
{
  if (packet.kind == sensor_protocol::Packet::Kind::Emg) {
    const auto &p = packet.emg;
    std::cout << "EMG id=" << static_cast<unsigned>(p.emg_id)
              << " seq=" << p.emg_seq
              << " sample=" << p.sample_index
              << " mega_us=" << p.mega_time_us
              << " grid_us=" << p.grid_time_us
              << " raw=[" << p.emg_raw[0] << ',' << p.emg_raw[1] << ','
              << p.emg_raw[2] << ',' << p.emg_raw[3] << "] valid=" << p.valid << '\n';
    return;
  }

  const auto &p = packet.imu;
  std::cout << "IMU id=" << static_cast<unsigned>(p.imu_id)
            << " seq=" << p.imu_seq
            << " sample=" << p.sample_index
            << " mega_us=" << p.mega_time_us
            << " grid_us=" << p.grid_time_us
            << " acc=[" << p.acceleration_g[0] << ',' << p.acceleration_g[1] << ','
            << p.acceleration_g[2] << "] gyro=[" << p.gyro_dps[0] << ','
            << p.gyro_dps[1] << ',' << p.gyro_dps[2] << "] angle=["
            << p.angle_deg[0] << ',' << p.angle_deg[1] << ',' << p.angle_deg[2]
            << "] quat=[" << p.quaternion[0] << ',' << p.quaternion[1] << ','
            << p.quaternion[2] << ',' << p.quaternion[3] << "] valid=" << p.valid << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    const int fd = openSerial(options.port, options.baud);
    sensor_protocol::Decoder decoder;
    sensor_protocol::Packet latest;
    bool have_latest = false;
    auto onPacket = [&](const sensor_protocol::Packet &packet) {
      latest = packet;
      have_latest = true;
      if (options.verbose) {
        printPacket(packet);
      }
    };

    const auto start = std::chrono::steady_clock::now();
    auto next_report = start + std::chrono::seconds(1);
    std::uint8_t bytes[4096];
    std::cout << "Listening on " << options.port << " @ " << options.baud
              << ". Ctrl-C to stop.\n";

    while (g_running) {
      pollfd descriptor{fd, POLLIN, 0};
      const int poll_result = ::poll(&descriptor, 1, 100);
      if (poll_result > 0 && (descriptor.revents & POLLIN)) {
        const ssize_t count = ::read(fd, bytes, sizeof(bytes));
        if (count > 0) {
          decoder.feed(bytes, static_cast<std::size_t>(count), onPacket);
        } else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
          throw std::runtime_error("serial read: " + std::string(std::strerror(errno)));
        }
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= next_report) {
        const auto &s = decoder.stats();
        std::cout << "stats bytes=" << s.bytes << " frames=" << s.frames
                  << " emg=" << s.emg_frames << " imu=" << s.imu_frames
                  << " crc_errors=" << s.crc_errors
                  << " malformed=" << s.malformed_frames
                  << " unknown=" << s.unknown_frames << '\n';
        if (!options.verbose && have_latest) {
          printPacket(latest);
        }
        next_report = now + std::chrono::seconds(1);
      }

      if (options.seconds > 0.0 &&
          std::chrono::duration<double>(now - start).count() >= options.seconds) {
        break;
      }
    }

    const auto &s = decoder.stats();
    std::cout << "final frames=" << s.frames << " emg=" << s.emg_frames
              << " imu=" << s.imu_frames << " crc_errors=" << s.crc_errors
              << " malformed=" << s.malformed_frames << '\n';
    ::close(fd);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "decoder_debug: " << error.what() << '\n';
    return 1;
  }
}
