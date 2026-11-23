# One HWT906P sample containing raw motion channels and module fusion output.
std_msgs/Header header

uint8 imu_id
uint32 imu_seq

uint32 sample_index
uint64 mega_time_us
uint64 grid_time_us

float32[3] acceleration_g
float32[3] gyro_dps
float32[3] angle_deg

# Fixed order: [w, x, y, z]
float32[4] quaternion

bool valid
