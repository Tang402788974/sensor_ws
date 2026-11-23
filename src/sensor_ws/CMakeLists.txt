cmake_minimum_required(VERSION 3.8)
project(sensor_ws LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(CTest)

if(DEFINED ENV{VIRTUAL_ENV} AND EXISTS "$ENV{VIRTUAL_ENV}/bin/python3")
  set(SENSOR_WS_PYARROW_PYTHON "$ENV{VIRTUAL_ENV}/bin/python3")
else()
  get_filename_component(SENSOR_WS_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  set(SENSOR_WS_PYARROW_PYTHON "${SENSOR_WS_WORKSPACE_ROOT}/.venv-sensor-ws/bin/python3")
  if(NOT EXISTS "${SENSOR_WS_PYARROW_PYTHON}")
    set(SENSOR_WS_PYARROW_PYTHON "${Python3_EXECUTABLE}")
  endif()
endif()

execute_process(
  COMMAND "${SENSOR_WS_PYARROW_PYTHON}" -c
    "import pathlib, pyarrow as pa; lib=pathlib.Path(pa.get_library_dirs()[0]); print(pa.get_include()); print(next(lib.glob('libarrow.so.*'))); print(next(lib.glob('libparquet.so.*')))"
  RESULT_VARIABLE SENSOR_WS_PYARROW_RESULT
  OUTPUT_VARIABLE SENSOR_WS_PYARROW_PATHS
  ERROR_VARIABLE SENSOR_WS_PYARROW_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT SENSOR_WS_PYARROW_RESULT EQUAL 0)
  message(FATAL_ERROR "PyArrow C++ headers/libraries are required. Activate .venv-sensor-ws first: ${SENSOR_WS_PYARROW_ERROR}")
endif()
string(REPLACE "\n" ";" SENSOR_WS_PYARROW_PATHS "${SENSOR_WS_PYARROW_PATHS}")
list(GET SENSOR_WS_PYARROW_PATHS 0 SENSOR_WS_PYARROW_INCLUDE)
list(GET SENSOR_WS_PYARROW_PATHS 1 SENSOR_WS_ARROW_LIBRARY)
list(GET SENSOR_WS_PYARROW_PATHS 2 SENSOR_WS_PARQUET_LIBRARY)
get_filename_component(SENSOR_WS_PYARROW_LIBRARY_DIR "${SENSOR_WS_ARROW_LIBRARY}" DIRECTORY)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EmgRaw.msg"
  "msg/ImuFeatures.msg"
  "msg/ImuJoint.msg"
  "msg/MonitorImuJoint.msg"
  "msg/VisionTheme.msg"
  DEPENDENCIES std_msgs
)

add_executable(decoder_debug src/decoder_debug.cpp)
target_include_directories(decoder_debug PRIVATE src)
target_compile_options(decoder_debug PRIVATE -Wall -Wextra -Wpedantic)

add_executable(sensor_data_pub src/sensor_data_pub.cpp)
target_include_directories(sensor_data_pub PRIVATE src)
target_compile_options(sensor_data_pub PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(sensor_data_pub rclcpp std_msgs)
add_dependencies(sensor_data_pub ${PROJECT_NAME}__rosidl_generator_cpp)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(sensor_data_pub "${cpp_typesupport_target}")

add_executable(imu_calib_zero src/imu_calib_zero.cpp)
target_include_directories(imu_calib_zero PRIVATE src)
target_compile_options(imu_calib_zero PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(imu_calib_zero rclcpp)
add_dependencies(imu_calib_zero ${PROJECT_NAME}__rosidl_generator_cpp)
target_link_libraries(imu_calib_zero "${cpp_typesupport_target}" nlohmann_json::nlohmann_json)

add_executable(monitor_joint_pub src/monitor_joint_pub.cpp)
target_include_directories(monitor_joint_pub PRIVATE src)
target_compile_options(monitor_joint_pub PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(monitor_joint_pub rclcpp)
add_dependencies(monitor_joint_pub ${PROJECT_NAME}__rosidl_generator_cpp)
target_link_libraries(monitor_joint_pub "${cpp_typesupport_target}" nlohmann_json::nlohmann_json)

add_executable(vision_theme_pub src/vision_theme_pub.cpp)
target_compile_options(vision_theme_pub PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(vision_theme_pub rclcpp rcl_interfaces std_msgs)
add_dependencies(vision_theme_pub ${PROJECT_NAME}__rosidl_generator_cpp)
target_link_libraries(vision_theme_pub "${cpp_typesupport_target}")

add_executable(gen_action_model src/gen_action_model.cpp)
target_include_directories(gen_action_model SYSTEM PRIVATE "${SENSOR_WS_PYARROW_INCLUDE}")
target_compile_options(gen_action_model PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gen_action_model
  nlohmann_json::nlohmann_json
  "${SENSOR_WS_PARQUET_LIBRARY}"
  "${SENSOR_WS_ARROW_LIBRARY}"
)
set_target_properties(gen_action_model PROPERTIES
  BUILD_RPATH "${SENSOR_WS_PYARROW_LIBRARY_DIR}"
  INSTALL_RPATH "${SENSOR_WS_PYARROW_LIBRARY_DIR}"
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_imu_quaternion test/test_imu_quaternion.cpp)
  target_include_directories(test_imu_quaternion PRIVATE src)
  target_compile_options(test_imu_quaternion PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS
  decoder_debug
  sensor_data_pub
  imu_calib_zero
  monitor_joint_pub
  vision_theme_pub
  gen_action_model
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY msg DESTINATION share/${PROJECT_NAME})
install(PROGRAMS test/imu_test_publisher.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/imu_vision_joint_convert.py DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
