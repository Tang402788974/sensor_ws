#pragma once

#include "imu_quaternion.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace sensor_ws {

using JointAngles = std::array<double, 7>;
using RotationMatrix = std::array<std::array<double, 3>, 3>;

inline Quaternion openarmMount(std::size_t segment) {
  constexpr double kHalf = 0.5;
  constexpr double kRootHalf = 0.70710678118654752440;
  if (segment == 0) return {kHalf, -kHalf, kHalf, -kHalf};
  return {0.0, 0.0, kRootHalf, -kRootHalf};
}

inline Quaternion relativeToOpenarmFrame(const Quaternion& relative,
                                         std::size_t proximal_segment,
                                         std::size_t distal_segment) {
  return normalized(multiply(
      multiply(openarmMount(proximal_segment), relative),
      conjugate(openarmMount(distal_segment))));
}

inline RotationMatrix rotationMatrix(const Quaternion& input) {
  const Quaternion q = normalized(input);
  return {{{1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            2.0 * (q.x * q.y - q.w * q.z),
            2.0 * (q.x * q.z + q.w * q.y)},
           {2.0 * (q.x * q.y + q.w * q.z),
            1.0 - 2.0 * (q.x * q.x + q.z * q.z),
            2.0 * (q.y * q.z - q.w * q.x)},
           {2.0 * (q.x * q.z - q.w * q.y),
            2.0 * (q.y * q.z + q.w * q.x),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y)}}};
}

inline JointAngles openarmJointAngles(const Quaternion& q01, const Quaternion& q12,
                                      const Quaternion& q23) {
  const auto shoulder = rotationMatrix(q01);
  const auto elbow = rotationMatrix(q12);
  const auto wrist = rotationMatrix(q23);
  const double shoulder_first = std::atan2(shoulder[0][2], shoulder[2][2]);
  const double shoulder_second = std::asin(std::clamp(-shoulder[1][2], -1.0, 1.0));
  const double shoulder_third = std::atan2(shoulder[1][0], shoulder[1][1]);
  const double elbow_first = std::atan2(elbow[0][2], elbow[2][2]);
  const double elbow_second = std::atan2(elbow[1][0], elbow[1][1]);
  const double wrist_first = std::atan2(-wrist[2][0], wrist[0][0]);
  const double wrist_second = std::atan2(-wrist[1][2], wrist[1][1]);
  return {-shoulder_first, -shoulder_second, -shoulder_third,
          -elbow_first, -elbow_second, wrist_first, wrist_second};
}

}  // namespace sensor_ws
