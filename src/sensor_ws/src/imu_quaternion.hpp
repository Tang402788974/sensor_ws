#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sensor_ws {

struct Quaternion {
  double w;
  double x;
  double y;
  double z;
};

struct EulerZyx {
  double yaw;
  double pitch;
  double roll;
};

inline Quaternion normalized(const Quaternion& q) {
  if (!std::isfinite(q.w) || !std::isfinite(q.x) || !std::isfinite(q.y) ||
      !std::isfinite(q.z)) {
    throw std::invalid_argument("quaternion contains a non-finite component");
  }
  const double norm = std::hypot(std::hypot(q.w, q.x), std::hypot(q.y, q.z));
  if (!std::isfinite(norm) || norm <= 1e-12) {
    throw std::invalid_argument("quaternion norm is too small");
  }
  return {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

inline Quaternion conjugate(const Quaternion& q) { return {q.w, -q.x, -q.y, -q.z}; }

inline Quaternion multiply(const Quaternion& a, const Quaternion& b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

inline Quaternion relative(const Quaternion& reference, const Quaternion& current) {
  return normalized(multiply(conjugate(normalized(reference)), normalized(current)));
}

inline Quaternion signCorrectedMean(const std::vector<Quaternion>& values) {
  if (values.empty()) throw std::invalid_argument("cannot average no quaternions");
  const Quaternion reference = normalized(values.front());
  Quaternion sum{0.0, 0.0, 0.0, 0.0};
  for (const auto& value : values) {
    const Quaternion q = normalized(value);
    const double dot = reference.w * q.w + reference.x * q.x +
                       reference.y * q.y + reference.z * q.z;
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    sum.w += sign * q.w;
    sum.x += sign * q.x;
    sum.y += sign * q.y;
    sum.z += sign * q.z;
  }
  return normalized(sum);
}

inline EulerZyx toEulerZyx(const Quaternion& input) {
  const Quaternion q = normalized(input);
  const double sin_pitch = std::clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0);
  return {std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
          std::asin(sin_pitch),
          std::atan2(2.0 * (q.w * q.x + q.y * q.z),
                     1.0 - 2.0 * (q.x * q.x + q.y * q.y))};
}

}  // namespace sensor_ws
