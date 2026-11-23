std_msgs/Header header

string session_id
string calibration_id

uint32 sample_index
uint64 grid_time_us
uint64 mega_time_us

uint32 theme_id
uint32 trial_id

# OpenArm v2 right-arm base coordinates, ordered J1 through J7, in radians.
float32[7] human_joint_ref_rad

bool human_joint_ref_valid

# -------- IMU 全局四元数 q_g[w,x,y,z] --------
float32[4] q_g0_wxyz
float32[4] q_g1_wxyz
float32[4] q_g2_wxyz
float32[4] q_g3_wxyz

# -------- IMU 相对姿态四元数 --------
float32[4] q_01
float32[4] q_12
float32[4] q_23

# -------- 零位增量四元数（标定后） --------
float32[4] q_01_delta
float32[4] q_12_delta
float32[4] q_23_delta

# -------- IMU 有效掩码 --------
bool imu_valid_0
bool imu_valid_1
bool imu_valid_2
bool imu_valid_3

# -------- 视觉标签附加信息(只保留先不用) --------
uint32 scene_id
float32[] scene_weights
float32[] theme_weights
