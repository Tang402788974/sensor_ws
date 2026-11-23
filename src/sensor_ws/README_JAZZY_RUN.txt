# Raspberry Pi 5 ROS 2 Jazzy: IMU calibration and dataset conversion

The action-model dataset uses IMU and vision data. EMG belongs to the separate
physiology model and is not required for this Parquet conversion.

## Install PyArrow and build

```bash
source /opt/ros/jazzy/setup.bash
cd /home/lylaff/lylaff
python3 -m venv --system-site-packages /home/lylaff/lylaff/.venv-sensor-ws
/home/lylaff/lylaff/.venv-sensor-ws/bin/pip install \
  -r /home/lylaff/lylaff/src/sensor_ws/requirements.txt
source /home/lylaff/lylaff/.venv-sensor-ws/bin/activate
colcon build --packages-select sensor_ws
source install/setup.bash
```

The package links its C++ Parquet reader to the Arrow libraries bundled with
PyArrow, so activate this environment before building.

## Calibrate the IMU zero pose

Before starting, place the right arm in the agreed zero pose: arm relaxed and
hanging down, thumb facing forward. Hold still until the node exits. The node
collects about 3 seconds of synchronized samples, requires at least 240 complete
frames, limits each IMU to 20% invalid messages, and checks gyro RMS against
5 deg/s. Any failure exits non-zero and keeps the previous JSON file unchanged.

Run the real calibration to the configured file:

```bash
ros2 run sensor_ws imu_calib_zero
```

For a software-only check, use two terminals. The test publisher produces four
stationary IMUs; this does not validate physical placement:

```bash
ros2 run sensor_ws imu_test_publisher.py
```

```bash
ros2 run sensor_ws imu_calib_zero --ros-args \
  -p calibration_file:=/tmp/imu_zero_q_test.json
```

Check the generated quaternion norms:

```bash
python3 -c 'import json, math; d=json.load(open("/tmp/imu_zero_q_test.json")); print({k: math.sqrt(sum(x*x for x in d[k])) for k in ("q_01_zero", "q_12_zero", "q_23_zero")})'
```

## Run the virtual vision publisher

Until the camera classifier is ready, the dummy node publishes labels using
the Pi ROS system clock. IDs are `T_01=1` (zero pose), `T_02=2` (elbow flexion),
and `T_03=3` (elbow extension); the scene is fixed to `S_01=1`:

```bash
ros2 run sensor_ws vision_theme_pub --ros-args \
  -p theme_id:=1 -p publish_rate_hz:=10 -p valid:=true
```

For dataset capture, switch the theme when the action starts. In another
terminal, run `ros2 param set /vision_theme_pub theme_id 2` for elbow flexion,
`ros2 param set /vision_theme_pub theme_id 3` for extension, and set it to `1`
when returning to the zero pose. Only IDs 1 through 3 are accepted. Fixed-time
cycling is useful for checking the message and conversion pipeline, but its
boundaries do not reliably match the person's actual movements, so do not use
it for training labels.

## Record a training session

Make sure the Pi uses ROS system time (`use_sim_time` is false), then record the
IMU and synchronized vision topics together:

```bash
ros2 bag record -o /home/lylaff/lylaff/bags/arm-session-001 \
  /imu/raw /vision/theme
```

`/vision/theme.header.stamp` must be the image capture time on the Pi ROS clock.
The converter uses the latest event at or before the completed IMU sample's
latest Pi receive timestamp. An invalid event resets the previous label; a label
older than 200 ms is invalid. The same `sample_index` groups the four IMUs.

## Convert a bag

PyArrow is constrained below version 20 because the Pi's ROS Python environment
uses NumPy 1.26; PyArrow 26 requires NumPy 2.0. Activate the environment created
above before converting.

Convert a bag with the calibration used during that session:

```bash
python3 "$(ros2 pkg prefix sensor_ws)/lib/sensor_ws/imu_vision_joint_convert.py" \
  --bag /home/lylaff/lylaff/bags/arm-session-001 \
  --calibration /home/lylaff/lylaff/src/sensor_ws/config/imu_zero_q.json \
  --output /home/lylaff/lylaff/datasets/arm-session-001.parquet \
  --session-id arm-session-001 \
  --label-max-age-ms 200
```

The converter writes the fields from `msg/output.msg`. The Arrow `header` column
is a nested struct containing the latest IMU receive timestamp and its frame ID.
`trial_id` starts at 1 for each contiguous valid theme segment in a session;
invalid or expired labels use `theme_id=0, trial_id=0`. IMU validity is named
`imu_valid_0` through `imu_valid_3` by calibrated body segment. `scene_weights`
and `theme_weights` are empty because the current `/vision/theme` message does
not publish weight arrays. Invalid joint/quaternion vectors contain null values
and have their matching validity flags set false. The dataset has no prosthesis
state or prosthesis validity columns. Failed conversions leave an existing
output file untouched.

Human IMU mounting axes are transformed into the OpenArm v2 right-arm base
frame before joint angles are extracted. This uses the standard base convention
`x=forward, y=left, z=up` and the right-arm URDF axis sequence for joints 1
through 7. Existing calibration files remain valid because their zero
quaternions are stored in sensor-relative coordinates and transformed at
runtime.

The converter reads `storage_identifier` from the bag's `metadata.yaml`, so it
works with both MCAP and sqlite3 bags. Older bags that contain only `/imu/raw`
cannot produce valid vision labels; record both topics as shown above before
conversion.

## Monitor live joint references

The monitor combines all four `/imu/raw` IMUs by `sample_index`, applies the
same calibration and joint-axis conversion as the offline converter, and
publishes `sensor_ws/msg/MonitorImuJoint` on `/imu/joint`:

```bash
ros2 run sensor_ws monitor_joint_pub --ros-args \
  -p calibration_file:=/home/lylaff/lylaff/src/sensor_ws/config/imu_zero_q.json
ros2 topic echo /imu/joint
```

`human_joint_ref_rad` follows the OpenArm v2 joint-axis order in the mapped base
frame. The mapping is provisional and must be validated against known poses in
simulation before use for actuation.

## Generate the trajectory template

Pass all converted session files for averaging. Samples are split by
`(session_id, trial_id, theme_id)`, aligned to normalized action progress, and
averaged at 100 Hz by default:

```bash
ros2 run sensor_ws gen_action_model \
  --input /home/lylaff/lylaff/datasets/arm-session-002-v2.parquet \
  --output /home/lylaff/lylaff/src/sensor_ws/config/action_model.json \
  --sample-period-us 10000
```

Each template point contains only `grid_time_us`, `theme_id`, and
`prosthesis_joint_ref_rad[7]`. Template `grid_time_us` is elapsed time from the
start of that template, beginning at zero. The model records the OpenArm v2
right-arm base frame and URDF joint-axis order, but remains marked
`ready_for_actuation=false` until the dynamic mapping and mechanical limits are
verified. After changing the IMU-to-joint mapping, reconvert the bags to Parquet
and regenerate `action_model.json`; the generator rejects datasets without the
current coordinate-frame metadata. The available `arm-session-002`
recording contains only `T_01`; collect labeled `T_02` and `T_03` trials before
expecting those templates in the JSON.
