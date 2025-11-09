find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})
find_package(Threads REQUIRED)
find_package(ALSA REQUIRED)

# add_executable(go2_trajectory_follow go2_trajectory_follow.cpp)
# target_link_libraries(go2_trajectory_follow unitree_sdk2)

add_executable(go2_sport_client go2_sport_client.cpp)
target_link_libraries(go2_sport_client unitree_sdk2)

add_executable(go2_low_level go2_low_level.cpp)
target_link_libraries(go2_low_level unitree_sdk2)

add_executable(go2_stand_example go2_stand_example.cpp)
target_link_libraries(go2_stand_example unitree_sdk2)


add_executable(go2_robot_state_client go2_robot_state_client.cpp)
target_link_libraries(go2_robot_state_client unitree_sdk2)

add_executable(go2_video_client go2_video_client.cpp)
target_link_libraries(go2_video_client unitree_sdk2)

add_executable(go2_vui_client go2_vui_client.cpp)
target_link_libraries(go2_vui_client unitree_sdk2)

set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../assets/whisper.cpp)
set(WHISPER_BUILD_DIR ${WHISPER_DIR}/build)
include_directories(
    ${WHISPER_DIR}
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
)

add_executable(human_follow_control human_follow_control.cpp)

target_link_libraries(human_follow_control PRIVATE
    unitree_sdk2
    ${DDS_LIBS}
    ${OpenCV_LIBS}
    Threads::Threads
    ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
    ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
    ${ALSA_LIBRARIES}
    ${WHISPER_BUILD_DIR}/src/libwhisper.a
    ${WHISPER_BUILD_DIR}/ggml/src/libggml.a
    ${WHISPER_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${WHISPER_BUILD_DIR}/ggml/src/libggml-base.a
    pthread
    dl      
    m       
    gomp
)

target_include_directories(human_follow_control PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
    ${CMAKE_SOURCE_DIR}/thirdparty/include
    ${OpenCV_INCLUDE_DIRS}
    ${ALSA_INCLUDE_DIRS}
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
    /usr/local/include
)
# add_executable(custom_one_leg_controller custom_one_leg_controller.cpp)
# target_link_libraries(custom_one_leg_controller PRIVATE
#     unitree_sdk2
#     ${DDS_LIBS}
#     ${OpenCV_LIBS}
#     Threads::Threads
#     ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
#     ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
# )
# target_include_directories(custom_one_leg_controller PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
#     ${CMAKE_SOURCE_DIR}/thirdparty/include
#     ${OpenCV_INCLUDE_DIRS}
# )


# add_executable(realsense_person_follower realsense_person_follower.cpp)
#
# target_link_libraries(realsense_person_follower PRIVATE
#     # ROS2 libraries
#     ament_cmake::ament_cmake
#     rclcpp::rclcpp
#     image_transport::image_transport
#     cv_bridge::cv_bridge
#     sensor_msgs::sensor_msgs
#
#     # Other libraries
#     ${realsense2_LIBRARIES} # Still need headers, but not for linking
#     ${OpenCV_LIBS}
#     Threads::Threads
#     ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
#     ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
#     ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a
# )
#
# target_include_directories(realsense_person_follower PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
#     ${CMAKE_SOURCE_DIR}/thirdparty/include
# )


# add_executable(lidar_defaultcode lidar_defaultcode.cpp)
# target_include_directories(lidar_defaultcode PUBLIC
# 	# Unitree SDK paths
# 	${CMAKE_SOURCE_DIR}/include
# 	${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
# 	${CMAKE_SOURCE_DIR}/thirdparty/include
# 	# ROS 2 package paths
# 	${rclcpp_INCLUDE_DIRS}
# 	${sensor_msgs_INCLUDE_DIRS}
# )
# target_link_libraries(lidar_defaultcode PRIVATE
# 	# Unitree SDK libs
# 	${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a
# 	${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
# 	${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
# 	Threads::Threads
# 	# ROS 2 package libs
# 	${rclcpp_LIBRARIES}
# 	${sensor_msgs_LIBRARIES}
# )

add_executable(enhanced_follower enhanced_follower.cpp)

target_include_directories(enhanced_follower PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/eigen3  # For AEKF
)

target_link_libraries(enhanced_follower
    ${OpenCV_LIBS}
    unitree_sdk2  # Unitree SDK library
)
target_compile_options(enhanced_follower PRIVATE -std=c++17 -O3)
