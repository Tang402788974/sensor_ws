#pragma once

// Decoder for the binary stream emitted by mega2560_flash/raw/raw.ino.
// C++17, no ROS dependency: both the standalone debugger and the ROS node use it.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace sensor_protocol {

constexpr std::uint8_t kMagic0 = 0xA5;
constexpr std::uint8_t kMagic1 = 0x5A;
constexpr std::uint8_t kVersion = 3;
constexpr std::uint8_t kPacketEmg = 1;
constexpr std::uint8_t kPacketImu = 2;
constexpr std::uint8_t kPacketSensor = 3;
constexpr std::size_t kMaxPayload = 180;

static_assert(sizeof(float) == 4, "This protocol requires 32-bit float");

struct EmgPacket {
  std::uint8_t emg_id = 0;
  std::uint32_t emg_seq = 0;
  std::uint32_t sample_index = 0;
  std::uint64_t mega_time_us = 0;
  std::uint64_t grid_time_us = 0;
  float emg_raw[4] = {};
  bool valid = false;
};

struct ImuPacket {
  std::uint8_t imu_id = 0;
  std::uint32_t imu_seq = 0;
  std::uint32_t sample_index = 0;
  std::uint64_t mega_time_us = 0;
  std::uint64_t grid_time_us = 0;
  float acceleration_g[3] = {};
  float gyro_dps[3] = {};
  float angle_deg[3] = {};
  float quaternion[4] = {};  // [w, x, y, z]
  bool valid = false;
};

struct Packet {
  enum class Kind { Emg, Imu } kind = Kind::Emg;
  EmgPacket emg;
  ImuPacket imu;
};

struct Statistics {
  std::uint64_t bytes = 0;
  std::uint64_t frames = 0;
  std::uint64_t emg_frames = 0;
  std::uint64_t imu_frames = 0;
  std::uint64_t crc_errors = 0;
  std::uint64_t malformed_frames = 0;
  std::uint64_t unknown_frames = 0;
  std::uint64_t startup_drops = 0;
};

inline std::uint16_t crc16Ccitt(const std::uint8_t *data, std::size_t length)
{
  std::uint16_t crc = 0xFFFF;
  for (std::size_t i = 0; i < length; ++i) {
    crc ^= static_cast<std::uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000u) ? static_cast<std::uint16_t>((crc << 1) ^ 0x1021u)
                             : static_cast<std::uint16_t>(crc << 1);
    }
  }
  return crc;
}

class Decoder {
public:
  using Callback = std::function<void(const Packet &)>;

  void feed(const std::uint8_t *data, std::size_t length, const Callback &callback)
  {
    stats_.bytes += length;
    buffer_.insert(buffer_.end(), data, data + length);
    parse(callback);
  }

  const Statistics &stats() const { return stats_; }

private:
  static std::uint16_t u16(const std::uint8_t *p)
  {
    return static_cast<std::uint16_t>(p[0]) |
           (static_cast<std::uint16_t>(p[1]) << 8);
  }

  static std::uint32_t u32(const std::uint8_t *p)
  {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
  }

  static std::uint64_t u64(const std::uint8_t *p)
  {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return value;
  }

  static float f32(const std::uint8_t *p)
  {
    std::uint32_t bits = u32(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  static std::int16_t i16(const std::uint8_t *p)
  {
    return static_cast<std::int16_t>(u16(p));
  }

  static void decodeEmg(const std::uint8_t *p, EmgPacket &out)
  {
    out.emg_id = p[0];
    out.emg_seq = u32(p + 1);
    out.sample_index = u32(p + 5);
    out.mega_time_us = u64(p + 9);
    out.grid_time_us = u64(p + 17);
    for (int i = 0; i < 4; ++i) {
      out.emg_raw[i] = static_cast<float>(i16(p + 25 + 2 * i));
    }
    out.valid = p[33] != 0;
  }

  static void decodeImuRecord(const std::uint8_t *p, std::uint8_t imu_id,
                              std::uint32_t sample_index,
                              std::uint64_t mega_time_us, std::uint64_t grid_time_us,
                              ImuPacket &out)
  {
    out.imu_id = imu_id;
    // The synchronized sample index is the sequence for every IMU in the
    // aggregate frame; keeping four duplicate sequence fields off-wire saves
    // 16 bytes per frame.
    out.imu_seq = sample_index;
    out.sample_index = sample_index;
    out.mega_time_us = mega_time_us;
    out.grid_time_us = grid_time_us;
    for (int i = 0; i < 3; ++i) {
      out.acceleration_g[i] = static_cast<float>(i16(p + 2 * i)) * (16.0f / 32768.0f);
      out.gyro_dps[i] = static_cast<float>(i16(p + 6 + 2 * i)) * (2000.0f / 32768.0f);
      out.angle_deg[i] = static_cast<float>(i16(p + 12 + 2 * i)) * (180.0f / 32768.0f);
    }
    for (int i = 0; i < 4; ++i) {
      out.quaternion[i] = static_cast<float>(i16(p + 18 + 2 * i)) * (1.0f / 32768.0f);
    }
    out.valid = p[26] != 0;
  }

  void parse(const Callback &callback)
  {
    while (true) {
      if (buffer_.size() < 2) {
        return;
      }

      std::size_t magic = 0;
      while (magic + 1 < buffer_.size() &&
             (buffer_[magic] != kMagic0 || buffer_[magic + 1] != kMagic1)) {
        ++magic;
      }
      if (magic > 0) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(magic));
      }
      if (buffer_.size() < 2) {
        return;
      }
      if (buffer_[0] != kMagic0 || buffer_[1] != kMagic1) {
        buffer_.erase(buffer_.begin());
        continue;
      }
      if (buffer_.size() < 6) {
        return;
      }

      const std::uint8_t version = buffer_[2];
      const std::uint8_t type = buffer_[3];
      const std::uint16_t payload_length = u16(&buffer_[4]);
      if (version != kVersion || payload_length > kMaxPayload) {
        if (synchronized_) {
          ++stats_.malformed_frames;
        } else {
          ++stats_.startup_drops;
        }
        buffer_.erase(buffer_.begin());
        continue;
      }

      const std::size_t frame_length = 2 + 1 + 1 + 2 + payload_length + 2;
      if (buffer_.size() < frame_length) {
        return;
      }

      const std::uint16_t expected = crc16Ccitt(&buffer_[2], 4 + payload_length);
      const std::uint16_t received = u16(&buffer_[6 + payload_length]);
      if (expected != received) {
        if (synchronized_) {
          ++stats_.crc_errors;
        } else {
          ++stats_.startup_drops;
        }
        buffer_.erase(buffer_.begin());
        continue;
      }
      synchronized_ = true;

      const std::uint8_t *payload = &buffer_[6];
      Packet packet;
      if (type == kPacketEmg && payload_length == 34) {
        packet.kind = Packet::Kind::Emg;
        decodeEmg(payload, packet.emg);
        ++stats_.emg_frames;
        ++stats_.frames;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_length));
        if (callback) {
          callback(packet);
        }
        continue;
      } else if (type == kPacketSensor && payload_length == 142) {
        Packet emg_packet;
        emg_packet.kind = Packet::Kind::Emg;
        decodeEmg(payload, emg_packet.emg);
        ++stats_.emg_frames;
        if (callback) {
          callback(emg_packet);
        }

        const std::uint32_t sample_index = u32(payload + 5);
        const std::uint64_t mega_time_us = u64(payload + 9);
        const std::uint64_t grid_time_us = u64(payload + 17);
        ++stats_.frames;
        for (int i = 0; i < 4; ++i) {
          Packet imu_packet;
          imu_packet.kind = Packet::Kind::Imu;
          decodeImuRecord(payload + 34 + 27 * i, static_cast<std::uint8_t>(i + 1),
                          sample_index, mega_time_us, grid_time_us,
                          imu_packet.imu);
          ++stats_.imu_frames;
          if (callback) {
            callback(imu_packet);
          }
        }
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_length));
        continue;
      } else if (type == kPacketEmg || type == kPacketImu) {
        ++stats_.malformed_frames;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_length));
        continue;
      } else {
        ++stats_.unknown_frames;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_length));
        continue;
      }

      ++stats_.frames;
      buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_length));
      if (callback) {
        callback(packet);
      }
    }
  }

  std::vector<std::uint8_t> buffer_;
  Statistics stats_;
  bool synchronized_ = false;
};

}  // namespace sensor_protocol
