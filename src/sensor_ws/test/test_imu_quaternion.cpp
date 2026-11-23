#include "imu_quaternion.hpp"
#include "openarm_joint_mapping.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using sensor_ws::Quaternion;

constexpr double kPi = 3.14159265358979323846;

TEST(ImuQuaternion, IdentityAndNormalization) {
  const auto identity = sensor_ws::normalized({2.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(identity.w, 1.0);
  EXPECT_DOUBLE_EQ(identity.x, 0.0);
  EXPECT_DOUBLE_EQ(identity.y, 0.0);
  EXPECT_DOUBLE_EQ(identity.z, 0.0);
}

TEST(ImuQuaternion, MultiplicationAndInverse) {
  const Quaternion q{0.5, 0.5, 0.5, 0.5};
  const auto product = sensor_ws::multiply(q, sensor_ws::conjugate(q));
  EXPECT_NEAR(product.w, 1.0, 1e-12);
  EXPECT_NEAR(product.x, 0.0, 1e-12);
  EXPECT_NEAR(product.y, 0.0, 1e-12);
  EXPECT_NEAR(product.z, 0.0, 1e-12);
}

TEST(ImuQuaternion, NinetyDegreeAxisRotations) {
  const double h = std::sqrt(0.5);
  const auto x = sensor_ws::toEulerZyx({h, h, 0.0, 0.0});
  const auto y = sensor_ws::toEulerZyx({h, 0.0, h, 0.0});
  const auto z = sensor_ws::toEulerZyx({h, 0.0, 0.0, h});
  EXPECT_NEAR(x.roll, kPi / 2.0, 1e-12);
  EXPECT_NEAR(y.pitch, kPi / 2.0, 1e-12);
  EXPECT_NEAR(z.yaw, kPi / 2.0, 1e-12);
}

TEST(ImuQuaternion, SignCorrectedMeanTreatsOppositeSignsAsSameRotation) {
  const auto mean = sensor_ws::signCorrectedMean(
      std::vector<Quaternion>{{1.0, 0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0, 0.0}});
  EXPECT_NEAR(mean.w, 1.0, 1e-12);
  EXPECT_NEAR(mean.x, 0.0, 1e-12);
}

TEST(ImuQuaternion, ExtractsZyxOrderAndZeroDelta) {
  const auto angles = sensor_ws::toEulerZyx(
      {std::cos(kPi / 4.0), 0.0, 0.0, std::sin(kPi / 4.0)});
  EXPECT_NEAR(angles.yaw, kPi / 2.0, 1e-12);
  const auto delta = sensor_ws::relative({0.5, 0.5, 0.5, 0.5},
                                         {0.5, 0.5, 0.5, 0.5});
  EXPECT_NEAR(delta.w, 1.0, 1e-12);
  EXPECT_NEAR(delta.x, 0.0, 1e-12);
  EXPECT_NEAR(delta.y, 0.0, 1e-12);
  EXPECT_NEAR(delta.z, 0.0, 1e-12);
}

TEST(OpenArmJointMapping, ConvertsSensorMountsAndJointAxes) {
  const std::array<double, 7> expected{0.2, -0.25, 0.3, -0.35, 0.4, -0.45, 0.5};
  const auto rotation = [](double x, double y, double z, double angle) {
    const double sine = std::sin(angle / 2.0);
    return Quaternion{std::cos(angle / 2.0), x * sine, y * sine, z * sine};
  };
  const auto shoulder = sensor_ws::normalized(sensor_ws::multiply(
      sensor_ws::multiply(rotation(0.0, 1.0, 0.0, -expected[0]),
                          rotation(1.0, 0.0, 0.0, -expected[1])),
      rotation(0.0, 0.0, 1.0, -expected[2])));
  const auto elbow = sensor_ws::normalized(sensor_ws::multiply(
      rotation(0.0, 1.0, 0.0, -expected[3]),
      rotation(0.0, 0.0, 1.0, -expected[4])));
  const auto wrist = sensor_ws::normalized(sensor_ws::multiply(
      rotation(0.0, 1.0, 0.0, expected[5]),
      rotation(1.0, 0.0, 0.0, expected[6])));
  const auto imu0_mount = sensor_ws::openarmMount(0);
  const auto arm_mount = sensor_ws::openarmMount(1);
  const auto sensor_relative = [](const Quaternion& q, const Quaternion& proximal,
                                  const Quaternion& distal) {
    return sensor_ws::normalized(sensor_ws::multiply(
        sensor_ws::multiply(sensor_ws::conjugate(proximal), q), distal));
  };
  const auto actual = sensor_ws::openarmJointAngles(
      sensor_ws::relativeToOpenarmFrame(
          sensor_relative(shoulder, imu0_mount, arm_mount), 0, 1),
      sensor_ws::relativeToOpenarmFrame(
          sensor_relative(elbow, arm_mount, arm_mount), 1, 2),
      sensor_ws::relativeToOpenarmFrame(
          sensor_relative(wrist, arm_mount, arm_mount), 2, 3));
  for (std::size_t joint = 0; joint < expected.size(); ++joint) {
    EXPECT_NEAR(actual[joint], expected[joint], 1e-9);
  }
}

}  // namespace
