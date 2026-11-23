/*
  raw.ino - Arduino Mega 2560, 4-channel raw EMG + 4 x HWT906P

  Hardware
  --------
  USB/Serial0 -> Raspberry Pi 5
  IMU 1       -> Serial1 RX1 (D19)
  IMU 2       -> Serial2 RX2 (D17)
  IMU 3       -> Serial3 RX3 (D15)
  IMU 4       -> I2C SDA D20, SCL D21, address 0x50
  EMG         -> A0, A1, A2, A3

  HWT906P UART frames are WIT frames: 0x55 + type + 8 bytes + checksum.
  This firmware consumes 0x51 (acc), 0x52 (gyro), 0x53 (angle), and
  0x59 (quaternion). Enable all four output types in the HWT configuration
  software and set all UART sensors to 115200 baud and 100 Hz.

  Output is binary little-endian packets. Do not print text on Serial0.
  Packet format:
    A5 5A | version:u8 | type:u8 | payload_len:u16 | payload | crc16:u16
  CRC16-CCITT is calculated over version, type, payload_len, and payload.

  Payload type 3 (one synchronized sensor frame, 142 bytes):
    emg_id:u8, emg_seq:u32, sample_index:u32,
    mega_time_us:u64, grid_time_us:u64,
    emg_raw[4]:int16 (ADC counts), emg_valid:u8,
    four IMU records (acceleration[3]:int16,
    gyro[3]:int16, angle[3]:int16, quaternion[w,x,y,z]:int16[4], valid:u8).
    IMU IDs are record index + 1. The Raspberry Pi decoder converts the raw
    values to float32 engineering units and publishes five ROS messages.

  Time semantics:
    mega_time_us = extended Mega micros() time (actual local clock).
    grid_time_us = sample_index * 10000 us on the 100 Hz Timer1 grid.
    sample_index is the common Mega grid counter. It wraps as uint32_t.
    emg_seq counts emitted aggregate packets. IMU sequence values are
    derived from the common sample_index by the Raspberry Pi decoder.

  Notes:
    - This build samples the four raw ADC channels at the requested 100 Hz.
      A 1000 Hz raw-EMG build needs a higher-throughput link or block packets.
    - Mega ADC channels are scanned sequentially, not simultaneously.
    - The fourth sensor must actually expose HWT I2C registers. If it is a
      UART-only board, use another hardware UART instead of the I2C section.
*/

#include <Arduino.h>
#include <Wire.h>
#include <avr/interrupt.h>
#include <stdint.h>
#include <string.h>

// ------------------------- User configuration -------------------------

// The Mega 2560 USB CDC link is reliable at 1,000,000 baud with this packet.
// The Raspberry Pi decoder and launch scripts use the same value.
static const uint32_t USB_BAUD = 1000000UL;
static const uint32_t IMU_UART_BAUD = 115200UL;

static const uint8_t EMG_COUNT = 4;
static const uint8_t EMG_PINS[EMG_COUNT] = {A0, A1, A2, A3};

static const uint8_t IMU_COUNT = 4;
static const uint8_t IMU4_I2C_ADDRESS = 0x50;

static const uint32_t GRID_RATE_HZ = 100UL;
static const uint32_t EMG_RATE_HZ = 100UL;  // Four-channel EMG output rate; ADC runs at ~143 Hz.
static const uint32_t IMU_RATE_HZ = 100UL;
static const uint32_t GRID_PERIOD_US = 1000000UL / GRID_RATE_HZ;
static const uint32_t IMU_VALID_TIMEOUT_US = 30000UL;
// Sample the multiplexed EMG inputs at a phase that is independent of the
// 100 Hz output grid. The latest sample is copied into each 100 Hz frame.
static const uint32_t EMG_ADC_PERIOD_US = 7000UL;  // about 143 Hz

static_assert(GRID_RATE_HZ % EMG_RATE_HZ == 0, "EMG rate must divide grid rate");
static_assert(GRID_RATE_HZ % IMU_RATE_HZ == 0, "IMU rate must divide grid rate");

static const uint32_t EMG_DIV = GRID_RATE_HZ / EMG_RATE_HZ;
static const uint32_t IMU_DIV = GRID_RATE_HZ / IMU_RATE_HZ;

// HWT906P standard WIT output scales. Change these if the sensor range was changed.
static const float ACC_SCALE_G = 16.0f / 32768.0f;
static const float GYRO_SCALE_DPS = 2000.0f / 32768.0f;
static const float ANGLE_SCALE_DEG = 180.0f / 32768.0f;
static const float QUAT_SCALE = 1.0f / 32768.0f;

static const uint8_t PACKET_EMG = 1;
static const uint8_t PACKET_IMU = 2;
static const uint8_t PACKET_SENSOR = 3;
static const uint8_t PROTOCOL_VERSION = 3;
static const uint8_t PACKET_MAGIC_0 = 0xA5;
static const uint8_t PACKET_MAGIC_1 = 0x5A;
static const uint16_t MAX_PAYLOAD = 180;

// ------------------------- Timer and clock -----------------------------

struct ImuState;

volatile uint32_t g_grid_ticks = 0;

ISR(TIMER1_COMPA_vect)
{
  ++g_grid_ticks;
}

static void setupGridTimer()
{
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // 16 MHz / 8 / 1000 Hz - 1 = 1999 on the Mega 2560.
  OCR1A = (uint16_t)((F_CPU / 8UL / GRID_RATE_HZ) - 1UL);
  TCCR1B |= (1 << WGM12);  // CTC mode.
  TCCR1B |= (1 << CS11);   // prescaler 8.
  TIMSK1 |= (1 << OCIE1A);
  interrupts();
}

// Extend the 32-bit micros() counter across its approximately 70-minute wrap.
static uint64_t megaMicros64()
{
  static uint32_t last_raw = 0;
  static uint64_t high = 0;
  const uint32_t raw = micros();
  if (raw < last_raw) {
    high += (1ULL << 32);
  }
  last_raw = raw;
  return high | (uint64_t)raw;
}

// ------------------------- Sensor state --------------------------------

struct ImuState
{
  int16_t raw_acc[3];
  int16_t raw_gyro[3];
  int16_t raw_angle[3];
  int16_t raw_quat[4];
  float acc[3];
  float gyro[3];
  float angle[3];
  float quat[4];  // fixed order: w, x, y, z
  uint8_t field_mask;
  uint64_t partial_start_us;
  uint64_t last_group_us;
  bool group_valid;
};

struct WitParser
{
  uint8_t frame[11];
  uint8_t count;
};

ImuState g_imu[IMU_COUNT];
WitParser g_uart_parser[3];
uint32_t g_emg_seq = 0;
int16_t g_emg_latest[EMG_COUNT] = {};
uint32_t g_next_emg_sample_us = 0;

static int16_t readI16LE(const uint8_t *p)
{
  const uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  return (int16_t)u;
}

static void resetPartialGroup(ImuState &s, uint64_t now_us)
{
  s.field_mask = 0;
  s.partial_start_us = now_us;
}

static void applyWitFrame(uint8_t imu_index, const uint8_t *frame, uint64_t now_us)
{
  ImuState &s = g_imu[imu_index];

  // A complete 0x51/0x52/0x53/0x59 group normally arrives in a few ms.
  // Drop an incomplete group before combining it with a new cycle.
  if (s.field_mask != 0 && now_us - s.partial_start_us > 50000ULL) {
    resetPartialGroup(s, now_us);
  }
  if (s.field_mask == 0) {
    s.partial_start_us = now_us;
  }

  const uint8_t type = frame[1];
  if (type == 0x51) {
    for (uint8_t i = 0; i < 3; ++i) {
      s.raw_acc[i] = readI16LE(&frame[2 + 2 * i]);
      s.acc[i] = (float)s.raw_acc[i] * ACC_SCALE_G;
    }
    s.field_mask |= 0x01;
  } else if (type == 0x52) {
    for (uint8_t i = 0; i < 3; ++i) {
      s.raw_gyro[i] = readI16LE(&frame[2 + 2 * i]);
      s.gyro[i] = (float)s.raw_gyro[i] * GYRO_SCALE_DPS;
    }
    s.field_mask |= 0x02;
  } else if (type == 0x53) {
    for (uint8_t i = 0; i < 3; ++i) {
      s.raw_angle[i] = readI16LE(&frame[2 + 2 * i]);
      s.angle[i] = (float)s.raw_angle[i] * ANGLE_SCALE_DEG;
    }
    s.field_mask |= 0x04;
  } else if (type == 0x59) {
    for (uint8_t i = 0; i < 4; ++i) {
      s.raw_quat[i] = readI16LE(&frame[2 + 2 * i]);
      s.quat[i] = (float)s.raw_quat[i] * QUAT_SCALE;
    }
    s.field_mask |= 0x08;
  } else {
    return;
  }

  if (s.field_mask == 0x0F) {
    s.group_valid = true;
    s.last_group_us = now_us;
    s.field_mask = 0;
  }
}

static bool witChecksumOk(const uint8_t *frame)
{
  uint8_t sum = 0;
  for (uint8_t i = 0; i < 10; ++i) {
    sum = (uint8_t)(sum + frame[i]);
  }
  return sum == frame[10];
}

static void pollUartImu(uint8_t imu_index, HardwareSerial &port, uint64_t now_us)
{
  WitParser &p = g_uart_parser[imu_index];
  while (port.available() > 0) {
    const uint8_t b = (uint8_t)port.read();

    if (p.count == 0) {
      if (b != 0x55) {
        continue;
      }
    }
    p.frame[p.count++] = b;

    if (p.count == 2) {
      const uint8_t type = p.frame[1];
      if (type != 0x51 && type != 0x52 && type != 0x53 && type != 0x59) {
        p.count = 0;
        continue;
      }
    }

    if (p.count == sizeof(p.frame)) {
      if (witChecksumOk(p.frame)) {
        applyWitFrame(imu_index, p.frame, now_us);
      }
      p.count = 0;
    }
  }
}

// ------------------------- IMU 4 over I2C -------------------------------

static bool readI2CRegisters(uint8_t reg, uint8_t *dst, uint8_t length)
{
  Wire.beginTransmission(IMU4_I2C_ADDRESS);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t received = (uint8_t)Wire.requestFrom(IMU4_I2C_ADDRESS, length);
  for (uint8_t i = 0; i < received; ++i) {
    dst[i] = (uint8_t)Wire.read();
  }
  while (Wire.available() > 0) {
    (void)Wire.read();
  }
  return received == length;
}

static void updateI2CImu(uint64_t now_us)
{
  ImuState &s = g_imu[3];
  uint8_t raw[24];
  uint8_t qraw[8];
  const bool main_ok = readI2CRegisters(0x34, raw, sizeof(raw));
  const bool quat_ok = readI2CRegisters(0x51, qraw, sizeof(qraw));

  if (main_ok) {
    for (uint8_t i = 0; i < 3; ++i) {
      s.raw_acc[i] = readI16LE(&raw[2 * i]);
      s.raw_gyro[i] = readI16LE(&raw[6 + 2 * i]);
      s.raw_angle[i] = readI16LE(&raw[18 + 2 * i]);
      s.acc[i] = (float)s.raw_acc[i] * ACC_SCALE_G;
      s.gyro[i] = (float)s.raw_gyro[i] * GYRO_SCALE_DPS;
      s.angle[i] = (float)s.raw_angle[i] * ANGLE_SCALE_DEG;
    }
  }
  if (quat_ok) {
    for (uint8_t i = 0; i < 4; ++i) {
      s.raw_quat[i] = readI16LE(&qraw[2 * i]);
      s.quat[i] = (float)s.raw_quat[i] * QUAT_SCALE;
    }
  }

  // I2C reads are one complete scheduled sample. Emit valid=false if either
  // register transaction failed, so a missing sensor is visible in the rosbag.
  s.group_valid = main_ok && quat_ok;
  s.last_group_us = now_us;
}

// ------------------------- Binary packet writer ------------------------

static uint16_t crc16Ccitt(const uint8_t *data, uint16_t length)
{
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < length; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void putU8(uint8_t *dst, uint16_t &pos, uint8_t value)
{
  dst[pos++] = value;
}

static void putI16(uint8_t *dst, uint16_t &pos, int16_t value)
{
  const uint16_t bits = (uint16_t)value;
  dst[pos++] = (uint8_t)(bits & 0xFF);
  dst[pos++] = (uint8_t)(bits >> 8);
}

static void putU32(uint8_t *dst, uint16_t &pos, uint32_t value)
{
  dst[pos++] = (uint8_t)(value & 0xFF);
  dst[pos++] = (uint8_t)((value >> 8) & 0xFF);
  dst[pos++] = (uint8_t)((value >> 16) & 0xFF);
  dst[pos++] = (uint8_t)((value >> 24) & 0xFF);
}

static void putU64(uint8_t *dst, uint16_t &pos, uint64_t value)
{
  for (uint8_t i = 0; i < 8; ++i) {
    dst[pos++] = (uint8_t)(value & 0xFF);
    value >>= 8;
  }
}

static void sendPacket(uint8_t type, const uint8_t *payload, uint16_t payload_len)
{
  uint8_t packet[2 + 1 + 1 + 2 + MAX_PAYLOAD + 2];
  uint16_t pos = 0;
  packet[pos++] = PACKET_MAGIC_0;
  packet[pos++] = PACKET_MAGIC_1;
  packet[pos++] = PROTOCOL_VERSION;
  packet[pos++] = type;
  packet[pos++] = (uint8_t)(payload_len & 0xFF);
  packet[pos++] = (uint8_t)((payload_len >> 8) & 0xFF);
  memcpy(&packet[pos], payload, payload_len);
  pos += payload_len;

  const uint16_t crc = crc16Ccitt(&packet[2], (uint16_t)(pos - 2));
  packet[pos++] = (uint8_t)(crc & 0xFF);
  packet[pos++] = (uint8_t)((crc >> 8) & 0xFF);
  Serial.write(packet, pos);
}

static void putImuRecord(uint8_t *payload, uint16_t &pos, uint8_t imu_index,
                         uint64_t packet_time_us)
{
  ImuState &s = g_imu[imu_index];
  const bool fresh = s.group_valid && (packet_time_us - s.last_group_us <= IMU_VALID_TIMEOUT_US);

  for (uint8_t i = 0; i < 3; ++i) {
    putI16(payload, pos, s.raw_acc[i]);
  }
  for (uint8_t i = 0; i < 3; ++i) {
    putI16(payload, pos, s.raw_gyro[i]);
  }
  for (uint8_t i = 0; i < 3; ++i) {
    putI16(payload, pos, s.raw_angle[i]);
  }
  for (uint8_t i = 0; i < 4; ++i) {
    putI16(payload, pos, s.raw_quat[i]);
  }
  putU8(payload, pos, fresh ? 1 : 0);
}

static void updateEmgSample()
{
  const uint32_t now_us = micros();
  if ((int32_t)(now_us - g_next_emg_sample_us) < 0) {
    return;
  }
  g_next_emg_sample_us = now_us + EMG_ADC_PERIOD_US;
  for (uint8_t i = 0; i < EMG_COUNT; ++i) {
    g_emg_latest[i] = (int16_t)analogRead(EMG_PINS[i]);
  }
}

static void emitSensorFrame(uint32_t sample_index, uint64_t grid_time_us)
{
  uint8_t payload[MAX_PAYLOAD];
  uint16_t pos = 0;
  const uint64_t packet_time_us = megaMicros64();

  putU8(payload, pos, 0);  // emg_id 0 = the four-channel aggregate.
  putU32(payload, pos, g_emg_seq++);
  putU32(payload, pos, sample_index);
  putU64(payload, pos, packet_time_us);
  putU64(payload, pos, grid_time_us);
  for (uint8_t i = 0; i < EMG_COUNT; ++i) {
    putI16(payload, pos, g_emg_latest[i]);
  }
  putU8(payload, pos, 1);
  for (uint8_t i = 0; i < IMU_COUNT; ++i) {
    putImuRecord(payload, pos, i, packet_time_us);
  }
  sendPacket(PACKET_SENSOR, payload, pos);
}

// ------------------------- Arduino entry points ------------------------

void setup()
{
  for (uint8_t i = 0; i < IMU_COUNT; ++i) {
    memset(&g_imu[i], 0, sizeof(g_imu[i]));
    g_imu[i].quat[0] = 1.0f;
  }
  memset(g_uart_parser, 0, sizeof(g_uart_parser));

  Serial.begin(USB_BAUD);
  Serial1.begin(IMU_UART_BAUD);
  Serial2.begin(IMU_UART_BAUD);
  Serial3.begin(IMU_UART_BAUD);

  g_next_emg_sample_us = micros();
  updateEmgSample();

  Wire.begin();
  Wire.setClock(400000UL);
  setupGridTimer();
}

void loop()
{
  // Read UART bytes before doing scheduled work. HardwareSerial buffers the
  // bytes in interrupts, but polling here prevents its 64-byte RX buffers from
  // filling while a packet is being sent to the Raspberry Pi.
  uint64_t now_us = megaMicros64();
  updateEmgSample();
  pollUartImu(0, Serial1, now_us);
  pollUartImu(1, Serial2, now_us);
  pollUartImu(2, Serial3, now_us);

  uint32_t ticks;
  noInterrupts();
  ticks = g_grid_ticks;
  interrupts();

  static uint32_t processed_tick = 0;
  while (processed_tick != ticks) {
    ++processed_tick;
    const uint32_t sample_index = processed_tick;
    const uint64_t grid_time_us = (uint64_t)sample_index * GRID_PERIOD_US;

    if ((processed_tick % IMU_DIV) == 0) {
      now_us = megaMicros64();
      updateEmgSample();
      pollUartImu(0, Serial1, now_us);
      pollUartImu(1, Serial2, now_us);
      pollUartImu(2, Serial3, now_us);
      updateI2CImu(now_us);
      emitSensorFrame(sample_index, grid_time_us);
    }
  }
}
