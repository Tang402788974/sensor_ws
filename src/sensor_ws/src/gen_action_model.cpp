#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Json = nlohmann::json;
using Angles = std::array<double, 7>;

constexpr double kTwoPi = 6.28318530717958647692;
constexpr uint64_t kDefaultSamplePeriodUs = 10'000;
constexpr char kCoordinateFrame[] = "openarm_v2_right_arm_base";
constexpr char kJointAxisOrder[] = "J1=-Y,J2=-X,J3=-Z,J4=-Y,J5=-Z,J6=+Y,J7=+X";

struct Sample {
  uint64_t time_us;
  Angles angles;
};

using TrialKey = std::tuple<std::string, uint32_t, uint32_t>;
using Trials = std::map<TrialKey, std::vector<Sample>>;

struct Options {
  std::vector<std::string> inputs;
  std::string output;
  uint64_t sample_period_us = kDefaultSamplePeriodUs;
};

void check_status(const arrow::Status& status) {
  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--input") {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        options.inputs.emplace_back(argv[++i]);
      }
      if (options.inputs.empty()) {
        throw std::invalid_argument("--input requires at least one Parquet path");
      }
    } else if (argument == "--output") {
      if (++i >= argc) {
        throw std::invalid_argument("--output requires a JSON path");
      }
      options.output = argv[i];
    } else if (argument == "--sample-period-us") {
      if (++i >= argc) {
        throw std::invalid_argument("--sample-period-us requires an integer");
      }
      size_t parsed = 0;
      options.sample_period_us = std::stoull(argv[i], &parsed);
      if (parsed != std::string(argv[i]).size() || options.sample_period_us == 0) {
        throw std::invalid_argument("--sample-period-us must be a positive integer");
      }
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: gen_action_model --input SESSION.parquet [MORE.parquet ...] "
                   "--output action_model.json [--sample-period-us 10000]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.inputs.empty() || options.output.empty()) {
    throw std::invalid_argument("--input and --output are required");
  }
  return options;
}

std::shared_ptr<arrow::Table> read_parquet(const std::string& path) {
  auto file_result = arrow::io::ReadableFile::Open(path);
  if (!file_result.ok()) {
    throw std::runtime_error("cannot open " + path + ": " + file_result.status().ToString());
  }
  auto reader_result = parquet::arrow::OpenFile(*file_result, arrow::default_memory_pool());
  if (!reader_result.ok()) {
    throw std::runtime_error("cannot read " + path + ": " + reader_result.status().ToString());
  }
  std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);
  std::shared_ptr<arrow::Table> table;
  check_status(reader->ReadTable(&table));
  auto combined_result = table->CombineChunks();
  if (!combined_result.ok()) {
    throw std::runtime_error("cannot combine Parquet chunks: " + combined_result.status().ToString());
  }
  return *combined_result;
}

template <typename ArrayType>
std::shared_ptr<ArrayType> column_array(const std::shared_ptr<arrow::Table>& table,
                                        const std::string& name) {
  const auto column = table->GetColumnByName(name);
  if (!column || column->num_chunks() != 1) {
    throw std::runtime_error("Parquet column missing or not combined: " + name);
  }
  auto values = std::dynamic_pointer_cast<ArrayType>(column->chunk(0));
  if (!values) {
    throw std::runtime_error("Parquet column has wrong type: " + name);
  }
  return values;
}

void read_dataset(const std::string& path, Trials* trials, std::set<std::string>* sessions) {
  const auto table = read_parquet(path);
  if (table->num_rows() == 0) {
    throw std::runtime_error("Parquet dataset has no rows: " + path);
  }
  const auto metadata = table->schema()->metadata();
  const int coordinate_frame_index = metadata ? metadata->FindKey("coordinate_frame") : -1;
  const int schema_version_index = metadata ? metadata->FindKey("schema_version") : -1;
  const int joint_axis_order_index = metadata ? metadata->FindKey("joint_axis_order") : -1;
  if (coordinate_frame_index < 0 ||
      metadata->value(coordinate_frame_index) != kCoordinateFrame ||
      schema_version_index < 0 || metadata->value(schema_version_index) != "3" ||
      joint_axis_order_index < 0 || metadata->value(joint_axis_order_index) != kJointAxisOrder) {
    throw std::runtime_error(
        "Parquet coordinate frame or joint-axis mapping is missing or outdated; reconvert the bag with the current "
        "imu_vision_joint_convert.py: " + path);
  }

  const auto session_ids = column_array<arrow::StringArray>(table, "session_id");
  const auto trial_ids = column_array<arrow::UInt32Array>(table, "trial_id");
  const auto theme_ids = column_array<arrow::UInt32Array>(table, "theme_id");
  const auto timestamps = column_array<arrow::UInt64Array>(table, "grid_time_us");
  const auto joint_valid = column_array<arrow::BooleanArray>(table, "human_joint_ref_valid");
  const auto joints = column_array<arrow::FixedSizeListArray>(table, "human_joint_ref_rad");
  const auto joint_values = std::dynamic_pointer_cast<arrow::FloatArray>(joints->values());
  if (!joint_values || joints->value_length() != 7) {
    throw std::runtime_error("human_joint_ref_rad must be float32[7]");
  }
  std::array<std::shared_ptr<arrow::BooleanArray>, 4> imu_valid;
  for (uint32_t segment = 0; segment < 4; ++segment) {
    imu_valid[segment] = column_array<arrow::BooleanArray>(table, "imu_valid_" + std::to_string(segment));
  }

  for (int64_t row = 0; row < table->num_rows(); ++row) {
    if (!session_ids->IsValid(row) || !trial_ids->IsValid(row) || !theme_ids->IsValid(row) ||
        !timestamps->IsValid(row) || !joint_valid->IsValid(row) || !joint_valid->Value(row)) {
      continue;
    }
    const uint32_t trial_id = trial_ids->Value(row);
    const uint32_t theme_id = theme_ids->Value(row);
    if (trial_id == 0 || theme_id == 0 || !joints->IsValid(row)) {
      continue;
    }
    bool valid_imus = true;
    for (const auto& valid : imu_valid) {
      valid_imus = valid_imus && valid->IsValid(row) && valid->Value(row);
    }
    if (!valid_imus) {
      continue;
    }

    Angles angles{};
    const int64_t offset = joints->value_offset(row);
    bool finite_angles = true;
    for (int joint = 0; joint < 7; ++joint) {
      if (!joint_values->IsValid(offset + joint)) {
        finite_angles = false;
        break;
      }
      angles[joint] = joint_values->Value(offset + joint);
      finite_angles = finite_angles && std::isfinite(angles[joint]);
    }
    if (!finite_angles) {
      continue;
    }

    const std::string session_id = session_ids->GetString(row);
    sessions->insert(session_id);
    (*trials)[TrialKey{session_id, trial_id, theme_id}].push_back(
        Sample{timestamps->Value(row), angles});
  }
}

double shortest_delta(double start, double end) {
  return std::remainder(end - start, kTwoPi);
}

void prepare_trial(std::vector<Sample>* samples, const Angles* reference_start) {
  std::sort(samples->begin(), samples->end(),
            [](const Sample& a, const Sample& b) { return a.time_us < b.time_us; });
  samples->erase(std::unique(samples->begin(), samples->end(),
                             [](const Sample& a, const Sample& b) {
                               return a.time_us == b.time_us;
                             }),
                 samples->end());
  if (samples->empty()) {
    return;
  }

  Angles previous_raw = samples->front().angles;
  Angles previous_unwrapped = previous_raw;
  for (size_t sample = 1; sample < samples->size(); ++sample) {
    for (size_t joint = 0; joint < 7; ++joint) {
      previous_unwrapped[joint] += shortest_delta(previous_raw[joint], (*samples)[sample].angles[joint]);
      previous_raw[joint] = (*samples)[sample].angles[joint];
      (*samples)[sample].angles[joint] = previous_unwrapped[joint];
    }
  }

  if (reference_start != nullptr) {
    for (size_t joint = 0; joint < 7; ++joint) {
      const double turns = std::round(((*reference_start)[joint] - samples->front().angles[joint]) / kTwoPi);
      const double offset = turns * kTwoPi;
      for (auto& sample : *samples) {
        sample.angles[joint] += offset;
      }
    }
  }
}

Angles interpolate(const std::vector<Sample>& samples, double time_us) {
  if (time_us <= static_cast<double>(samples.front().time_us)) {
    return samples.front().angles;
  }
  if (time_us >= static_cast<double>(samples.back().time_us)) {
    return samples.back().angles;
  }
  const auto right = std::lower_bound(samples.begin(), samples.end(), time_us,
                                      [](const Sample& sample, double time) {
                                        return static_cast<double>(sample.time_us) < time;
                                      });
  const auto left = right - 1;
  const double span = static_cast<double>(right->time_us - left->time_us);
  const double ratio = span > 0.0 ? (time_us - static_cast<double>(left->time_us)) / span : 0.0;
  Angles result{};
  for (size_t joint = 0; joint < 7; ++joint) {
    result[joint] = left->angles[joint] + ratio * (right->angles[joint] - left->angles[joint]);
  }
  return result;
}

uint64_t median_duration(std::vector<uint64_t> durations) {
  std::sort(durations.begin(), durations.end());
  const size_t middle = durations.size() / 2;
  if (durations.size() % 2 != 0) {
    return durations[middle];
  }
  return durations[middle - 1] / 2 + durations[middle] / 2 +
         (durations[middle - 1] % 2 + durations[middle] % 2) / 2;
}

std::string theme_name(uint32_t theme_id) {
  std::ostringstream name;
  name << "T_";
  if (theme_id < 10) {
    name << '0';
  }
  name << theme_id;
  return name.str();
}

Json build_model(const Trials& trials, const std::set<std::string>& sessions,
                 uint64_t sample_period_us) {
  std::map<uint32_t, std::vector<std::vector<Sample>>> trials_by_theme;
  for (const auto& [key, raw_samples] : trials) {
    const uint32_t theme_id = std::get<2>(key);
    trials_by_theme[theme_id].push_back(raw_samples);
  }

  Json templates = Json::array();
  for (auto& [theme_id, theme_trials] : trials_by_theme) {
    Angles reference_start{};
    bool has_reference_start = false;
    std::vector<std::vector<Sample>> prepared;
    for (auto& trial : theme_trials) {
      prepare_trial(&trial, has_reference_start ? &reference_start : nullptr);
      if (trial.empty()) {
        continue;
      }
      if (!has_reference_start) {
        reference_start = trial.front().angles;
        has_reference_start = true;
      }
      prepared.push_back(std::move(trial));
    }
    if (prepared.empty()) {
      continue;
    }

    std::vector<uint64_t> durations;
    durations.reserve(prepared.size());
    for (const auto& trial : prepared) {
      durations.push_back(trial.back().time_us - trial.front().time_us);
    }
    const uint64_t median = median_duration(durations);
    const uint64_t duration_us = (median / sample_period_us) * sample_period_us;
    const uint64_t point_count = duration_us / sample_period_us + 1;
    if (point_count > 10'000'000) {
      throw std::runtime_error("template is too long to resample safely");
    }

    Json samples = Json::array();
    for (uint64_t point = 0; point < point_count; ++point) {
      const uint64_t output_time_us = point * sample_period_us;
      const double phase = duration_us == 0 ? 0.0 :
          static_cast<double>(output_time_us) / static_cast<double>(duration_us);
      std::array<double, 7> totals{};
      for (const auto& trial : prepared) {
        const uint64_t trial_duration = trial.back().time_us - trial.front().time_us;
        const double source_time = static_cast<double>(trial.front().time_us) +
            phase * static_cast<double>(trial_duration);
        const Angles angles = interpolate(trial, source_time);
        for (size_t joint = 0; joint < 7; ++joint) {
          totals[joint] += angles[joint];
        }
      }
      Json prosthesis_target = Json::array();
      for (const double total : totals) {
        prosthesis_target.push_back(total / static_cast<double>(prepared.size()));
      }
      samples.push_back({
          {"grid_time_us", output_time_us},
          {"theme_id", theme_id},
          {"prosthesis_joint_ref_rad", std::move(prosthesis_target)},
      });
    }

    templates.push_back({
        {"theme_id", theme_id},
        {"theme_name", theme_name(theme_id)},
        {"trial_count", prepared.size()},
        {"duration_us", duration_us},
        {"samples", std::move(samples)},
    });
  }
  if (templates.empty()) {
    throw std::runtime_error("no valid labeled human joint trials found in input Parquet files");
  }

  return {
      {"schema_version", 2},
      {"angle_unit", "rad"},
      {"joint_order", {"J1", "J2", "J3", "J4", "J5", "J6", "J7"}},
      {"sample_period_us", sample_period_us},
      {"source_field", "human_joint_ref_rad"},
      {"output_field", "prosthesis_joint_ref_rad"},
      {"coordinate_frame", kCoordinateFrame},
      {"joint_mapping", "IMU axes transformed to OpenArm v2 right-arm base; URDF axis order J1-J7"},
      {"source_session_ids", sessions},
      {"is_provisional", true},
      {"ready_for_actuation", false},
      {"provisional_reasons", {"verify dynamic IMU-to-joint mapping and mechanical limits before actuation"}},
      {"templates", std::move(templates)},
  };
}

void write_model(const Json& model, const std::string& output_path) {
  const std::filesystem::path output(output_path);
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  const std::filesystem::path temporary = output.string() + ".tmp";
  {
    std::ofstream destination(temporary, std::ios::binary | std::ios::trunc);
    if (!destination) {
      throw std::runtime_error("cannot create output file: " + temporary.string());
    }
    destination << model.dump(2) << '\n';
    destination.flush();
    if (!destination) {
      throw std::runtime_error("failed to write output file: " + temporary.string());
    }
  }
  std::filesystem::rename(temporary, output);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Trials trials;
    std::set<std::string> sessions;
    for (const auto& input : options.inputs) {
      read_dataset(input, &trials, &sessions);
    }
    const Json model = build_model(trials, sessions, options.sample_period_us);
    write_model(model, options.output);
    std::cout << "Wrote " << model.at("templates").size() << " theme templates to "
              << options.output << '\n';
    std::cout << "Template trajectories are provisional; verify live joint mapping before actuation.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gen_action_model: " << error.what() << '\n';
    return 1;
  }
}
