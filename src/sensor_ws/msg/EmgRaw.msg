# One synchronized group of four raw EMG channels.
std_msgs/Header header

uint8 emg_id
uint32 emg_seq

uint32 sample_index
uint64 mega_time_us
uint64 grid_time_us

float32[4] emg_raw

bool valid
