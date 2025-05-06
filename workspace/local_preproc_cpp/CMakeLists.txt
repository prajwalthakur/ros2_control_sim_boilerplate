cmake_minimum_required(VERSION 3.8)
project(local_preproc_cpp)

# (optional) respect PCL_ROOT if set
cmake_policy(SET CMP0074 NEW)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# THIS line must pull in segmentation
message(STATUS ">>> PCL_INCLUDE_DIRS = ${PCL_INCLUDE_DIRS}")
message(STATUS ">>> PCL_LIBRARY_DIRS = ${PCL_LIBRARY_DIRS}")
message(STATUS ">>> PCL_LIBRARIES = ${PCL_LIBRARIES}")

# include PCL headers
include_directories(
  ${PCL_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(scan_preprocess src/scan_preprocess.cpp)

ament_target_dependencies(scan_preprocess
  rclcpp
  sensor_msgs
  std_msgs
  nav_msgs
  geometry_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

# here we actually link against all the PCL libs, including segmentation
target_link_libraries(scan_preprocess
  ${PCL_LIBRARIES}
)

install(TARGETS scan_preprocess
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
