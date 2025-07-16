
# add_subdirectory(helloworld)
# add_subdirectory(wireless_controller)
# add_subdirectory(jsonize)
# add_subdirectory(state_machine)


add_subdirectory(go2)
# add_subdirectory(b2)
# add_subdirectory(h1)
# add_subdirectory(g1)
# add_subdirectory(go2w)
# add_subdirectory(b2w)

add_executable(keyboard_control keyboard_control.cpp)
add_executable(sitonhindlegs sitonhindlegs.cpp)

target_include_directories(keyboard_control PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
	${CMAKE_SOURCE_DIR}/thirdparty/include
)
target_include_directories(sitonhindlegs PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/thirdparty/include/ddscxx
	${CMAKE_SOURCE_DIR}/thirdparty/include
)



find_package(Threads REQUIRED)
target_link_libraries(keyboard_control PRIVATE
	${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a
	${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so
	${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so
	Threads::Threads
)
target_link_libraries(sitonhindlegs PRIVATE
	${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a     # Unitree SDK
        ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so  # DDS C++ lib
        ${CMAKE_SOURCE_DIR}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so     # DDS C lib
        Threads::Threads
)

get_target_property(KC_INCLUDES keyboard_control INCLUDE_DIRECTORIES)
message(STATUS "Include directories for keyboard_control: ${KC_INCLUDES}")
