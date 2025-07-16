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

# Add the new hand gesture control executable
add_executable(hand_gesture_control hand_gesture_control.cpp)

target_link_libraries(hand_gesture_control PRIVATE
    unitree_sdk2
    ${DDS_LIBS}
    ${OpenCV_LIBS}
    Threads::Threads
    ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
    ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
)

target_include_directories(hand_gesture_control PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
    ${CMAKE_SOURCE_DIR}/thirdparty/include
    ${OpenCV_INCLUDE_DIRS}
)
