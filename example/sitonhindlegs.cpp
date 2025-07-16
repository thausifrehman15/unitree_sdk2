#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>       // For fabs, M_PI
#include <unistd.h>    // For usleep
#include <signal.h>    // For signal handling
#include <memory>      // For std::shared_ptr
#include <chrono>      // For timing
#include <atomic>      // For std::atomic_bool

// Unitree SDK Headers
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/common/time/time_tool.hpp"
#include "unitree/common/thread/thread.hpp"

#include "unitree/idl/go2/SportModeCmd_.hpp" // For SportModeCmd

// IDL Headers for LowCmd/LowState
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"

// Define topics
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_SPORTCMD "rt/sportmodecmd"

// Define constants
const float K_RAD_PER_DEG = M_PI / 180.0;
const int NUM_MOTORS = 12;
const int LOOP_FREQUENCY_HZ = 500; 
const useconds_t LOOP_SLEEP_US = 1000000 / LOOP_FREQUENCY_HZ; 
const float MOVE_DURATION_SEC = 10.0f; 
const float HOLD_DURATION_SEC = 5.0f;  
const float RETURN_DURATION_SEC = 10.0f; // Shortened return for quicker test

// Global variables
std::atomic_bool running(true); 
std::atomic_bool low_state_received(false); 
unitree_go::msg::dds_::LowState_ current_low_state{}; 
std::shared_ptr<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>> lowcmd_publisher;
std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>> lowstate_subscriber;
std::shared_ptr<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::SportModeCmd_>> sportcmd_publisher;

// Initial joint positions (rad)
std::vector<float> initial_q(NUM_MOTORS, 0.0f);

// Original target sit pose from high-level (for reference and individual joint targets)
std::vector<float> full_sit_target_q_ref = {
    0.0140111f,   1.44247f,    -1.03261f,    // FR_Hip, FR_Thigh, FR_Calf
   -0.0493753f,  1.47108f,    -1.0536f,     // FL_Hip, FL_Thigh, FL_Calf
    0.0234526f,   2.01676f,    -2.77505f,    // HR_Hip, HR_Thigh, HR_Calf
   -0.00356025f, 2.01798f,    -2.81018f     // HL_Hip, HL_Thigh, HL_Calf
};

// Target gains (can be tuned - STARTING with kp=100, kd=5 for this test)
float target_kp = 100.0f;
float target_kd = 5.0f;

// --- Function Declarations ---
uint32_t crc32_core(uint32_t* ptr, uint32_t len);
void LowStateMessageHandler(const void* msg);
void SigintHandler(int sig);
float Interpolate(float start, float end, float factor);
void SendLowCmd(const std::vector<float>& q_targets, float kp, float kd);
// Default arguments specified in the declaration:
void SendSportCmd(uint8_t mode, float body_height_target = 0.0f, float foot_raise_height_target = 0.08f);


// --- Main Function ---
int main(int argc, char** argv) {
    std::cout << "===================================================" << std::endl;
    std::cout << " Unitree Go2 Single Hind Leg Tuck Low-Level Example " << std::endl;
    std::cout << "===================================================" << std::endl << std::endl;

    if (argc > 1) {
        const char* commInterface = argv[1];
        std::cout << "User specified network interface: " << commInterface << std::endl;
        unitree::robot::ChannelFactory::Instance()->Init(0, commInterface); 
    } else {
        std::cout << "No network interface specified by user, using SDK default for Init." << std::endl;
        unitree::robot::ChannelFactory::Instance()->Init(0); // <<--- Use this overload
    }
    
    signal(SIGINT, SigintHandler);
    
    
    lowcmd_publisher = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(TOPIC_LOWCMD);
    lowcmd_publisher->InitChannel(); 
    std::cout << "LowCmd publisher initialized." << std::endl;

    sportcmd_publisher = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::SportModeCmd_>>(TOPIC_SPORTCMD);
    sportcmd_publisher->InitChannel(); 
    std::cout << "SportCmd publisher initialized on topic: " << TOPIC_SPORTCMD << std::endl;

    lowstate_subscriber = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(TOPIC_LOWSTATE);
    lowstate_subscriber->InitChannel(LowStateMessageHandler, 1); 
    std::cout << "LowState subscriber initialized." << std::endl;

    std::cout << "Waiting for initial robot state..." << std::endl;
    while (running.load() && !low_state_received.load()) {
        usleep(100000); 
    }

    if (!running.load()) {
        std::cout << "Exiting before receiving state." << std::endl;
        return 1; 
    }

    std::cout << "Initial joint positions received:" << std::endl;
    for(int i=0; i<NUM_MOTORS; ++i) {
        std::cout << "  Joint " << i << ": " << initial_q[i] << " rad" << std::endl;
    }
    std::cout << std::endl;

    std::vector<float> target_q_one_leg = initial_q; 

    target_q_one_leg[6] = full_sit_target_q_ref[6]; 
    target_q_one_leg[7] = full_sit_target_q_ref[7]; 
    target_q_one_leg[8] = full_sit_target_q_ref[8]; 
    
    std::cout << "Target joint positions for SINGLE HR LEG TUCK (others hold initial):" << std::endl;
    for(int i=0; i<NUM_MOTORS; ++i) {
        std::cout << "  Joint " << i << " Target: " << target_q_one_leg[i] << " rad (Initial: " << initial_q[i] << " rad)" << std::endl;
    }
    std::cout << std::endl;


    std::cout << "Commanding Sport Mode to Idle (mode 0) to enable low-level control..." << std::endl;
    SendSportCmd(0); 
    usleep(500000); 

    std::cout << "Robot will now attempt to tuck its HIND RIGHT leg." << std::endl;
    std::cout << "*** Robot will likely become UNSTABLE. Be ready! ***" << std::endl;
    std::cout << "Press Enter to continue or Ctrl+C to abort..." << std::endl;
    getchar(); 

    if (!running.load()) { 
        std::cout << "Aborting before motion." << std::endl;
        SendSportCmd(1); 
        usleep(500000);
        return 1; 
    }

    // --- Phase 1: Move HR leg to tuck ---
    std::cout << "Phase 1: Tucking HR leg (" << MOVE_DURATION_SEC << " seconds)..." << std::endl;
    auto phase1_start = std::chrono::steady_clock::now();
    std::chrono::duration<double> phase1_elapsed; 
    int log_counter = 0;
    const int log_interval = LOOP_FREQUENCY_HZ / 10; 

    while (running.load() && phase1_elapsed.count() < MOVE_DURATION_SEC) {
        phase1_elapsed = std::chrono::steady_clock::now() - phase1_start;
        float factor = phase1_elapsed.count() / MOVE_DURATION_SEC;
        factor = std::max(0.0f, std::min(1.0f, factor)); 

        std::vector<float> current_targets(NUM_MOTORS);
        for (int i = 0; i < NUM_MOTORS; ++i) {
            current_targets[i] = Interpolate(initial_q[i], target_q_one_leg[i], factor);
        }

        SendLowCmd(current_targets, target_kp, target_kd);

        if (++log_counter % log_interval == 0) {
            std::cout << "[P1: " << std::fixed << std::setprecision(2) << phase1_elapsed.count() << "s] ";
            std::cout << "HR Hip  T:" << std::fixed << std::setprecision(3) << current_targets[6] << " A:" << current_low_state.motor_state()[6].q() << " Tau:" << current_low_state.motor_state()[6].tau_est();
            std::cout << " | HR Thigh T:" << current_targets[7] << " A:" << current_low_state.motor_state()[7].q() << " Tau:" << current_low_state.motor_state()[7].tau_est();
            std::cout << " | HR Calf  T:" << current_targets[8] << " A:" << current_low_state.motor_state()[8].q() << " Tau:" << current_low_state.motor_state()[8].tau_est();
            std::cout << " | FootF HR:" << current_low_state.foot_force()[2] << " HL:" << current_low_state.foot_force()[3] << std::endl;
        }
        usleep(LOOP_SLEEP_US);
    }

    if (!running.load()) {
        std::cout << "Motion interrupted during Phase 1. Returning to stand..." << std::endl;
        SendSportCmd(1); 
        usleep(1000000);
        return 1; 
    }

    // --- Phase 2: Hold HR leg tucked ---
    std::cout << "Phase 2: Holding HR leg tucked (" << HOLD_DURATION_SEC << " seconds)..." << std::endl;
    auto phase2_start = std::chrono::steady_clock::now();
    std::chrono::duration<double> phase2_elapsed_p2; // Renamed to avoid conflict
    log_counter = 0;

    while (running.load() && phase2_elapsed_p2.count() < HOLD_DURATION_SEC) { // Used renamed variable
        phase2_elapsed_p2 = std::chrono::steady_clock::now() - phase2_start;

        SendLowCmd(target_q_one_leg, target_kp, target_kd);

        if (++log_counter % log_interval == 0) {
             std::cout << "[P2: " << std::fixed << std::setprecision(2) << phase2_elapsed_p2.count() << "s] ";
             std::cout << "HR Hip  A:" << std::fixed << std::setprecision(3) << current_low_state.motor_state()[6].q() << " Tau:" << current_low_state.motor_state()[6].tau_est();
             std::cout << " | HR Thigh A:" << current_low_state.motor_state()[7].q() << " Tau:" << current_low_state.motor_state()[7].tau_est();
             std::cout << " | HR Calf  A:" << current_low_state.motor_state()[8].q() << " Tau:" << current_low_state.motor_state()[8].tau_est();
             std::cout << " | FootF HR:" << current_low_state.foot_force()[2] << " HL:" << current_low_state.foot_force()[3] << std::endl;
        }
        usleep(LOOP_SLEEP_US);
    }

    if (!running.load()) {
        std::cout << "Motion interrupted during Phase 2. Returning to stand..." << std::endl;
        SendSportCmd(1); 
        usleep(1000000);
        return 1; 
    }

    // --- Phase 3: Return HR leg to initial position ---
    std::cout << "Phase 3: Returning HR leg to initial standing pose (" << RETURN_DURATION_SEC << " seconds)..." << std::endl;
    auto phase3_start = std::chrono::steady_clock::now();
    std::chrono::duration<double> phase3_elapsed_p3; // Renamed to avoid conflict
    log_counter = 0;

    std::vector<float> return_start_q_single_leg = target_q_one_leg; 

    while (running.load() && phase3_elapsed_p3.count() < RETURN_DURATION_SEC) { // Used renamed variable
        phase3_elapsed_p3 = std::chrono::steady_clock::now() - phase3_start;
        float factor = phase3_elapsed_p3.count() / RETURN_DURATION_SEC;
        factor = std::max(0.0f, std::min(1.0f, factor)); 

        std::vector<float> current_targets(NUM_MOTORS);
        for (int i = 0; i < NUM_MOTORS; ++i) {
            current_targets[i] = Interpolate(return_start_q_single_leg[i], initial_q[i], factor);
        }

        SendLowCmd(current_targets, target_kp, target_kd);

        if (++log_counter % log_interval == 0) {
             std::cout << "[P3: " << std::fixed << std::setprecision(2) << phase3_elapsed_p3.count() << "s] ";
             std::cout << "HR Hip  T:" << std::fixed << std::setprecision(3) << current_targets[6] << " A:" << current_low_state.motor_state()[6].q() << " Tau:" << current_low_state.motor_state()[6].tau_est();
             std::cout << " | HR Thigh T:" << current_targets[7] << " A:" << current_low_state.motor_state()[7].q() << " Tau:" << current_low_state.motor_state()[7].tau_est();
             std::cout << " | HR Calf  T:" << current_targets[8] << " A:" << current_low_state.motor_state()[8].q() << " Tau:" << current_low_state.motor_state()[8].tau_est();
             std::cout << " | FootF HR:" << current_low_state.foot_force()[2] << " HL:" << current_low_state.foot_force()[3] << std::endl;
        }
        usleep(LOOP_SLEEP_US);
    }

    if (!running.load()){ 
        std::cout << "Motion interrupted during Phase 3. Attempting to set to Sport Mode Stand..." << std::endl;
        SendSportCmd(1);
        usleep(1000000);
        return 1; 
    }

    std::cout << "Single HR leg tuck sequence finished." << std::endl;

    std::cout << "Commanding Sport Mode to Stand (mode 1)..." << std::endl;
    SendSportCmd(1); 
    usleep(1000000); 

    std::cout << "Robot should be attempting to stand." << std::endl;
    std::cout << "You can now use keyboard_control or the remote to ensure stable standing." << std::endl;

    return 0; 
}


// --- Function Definitions ---

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;
    for (unsigned int i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1; CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

void LowStateMessageHandler(const void* msg) {
    current_low_state = *(static_cast<const unitree_go::msg::dds_::LowState_*>(msg));
    if (!low_state_received.load()) {
        for (int i = 0; i < NUM_MOTORS; ++i) {
            initial_q[i] = current_low_state.motor_state()[i].q();
        }
        low_state_received.store(true); 
    }
}

void SigintHandler(int sig) {
    std::cout << "\nCtrl+C detected. Exiting." << std::endl;
    running.store(false);
}

float Interpolate(float start, float end, float factor) {
    factor = std::max(0.0f, std::min(1.0f, factor)); 
    return start + (end - start) * factor;
}

void SendLowCmd(const std::vector<float>& q_targets, float kp, float kd) {
    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0x00; 
    low_cmd.gpio() = 0;

    for (int i = 0; i < NUM_MOTORS; i++) {
        low_cmd.motor_cmd()[i].mode() = 0x01; 
        low_cmd.motor_cmd()[i].q() = q_targets[i];
        low_cmd.motor_cmd()[i].kp() = kp;
        low_cmd.motor_cmd()[i].kd() = kd;
        low_cmd.motor_cmd()[i].dq() = 0.0f;  
        low_cmd.motor_cmd()[i].tau() = 0.0f; 
    }

    low_cmd.crc() = crc32_core((uint32_t*)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) / 4) - 1);

    if (lowcmd_publisher) {
        lowcmd_publisher->Write(low_cmd);
    }
}

// No default arguments in the definition, only in the declaration:
void SendSportCmd(uint8_t mode, float body_height_target, float foot_raise_height_target) {
    if (!sportcmd_publisher) {
        std::cerr << "Error: sportcmd_publisher is not initialized! Cannot send SportModeCmd." << std::endl;
        return;
    }

    unitree_go::msg::dds_::SportModeCmd_ sport_cmd{}; 

    sport_cmd.mode() = mode;
    sport_cmd.gait_type() = 0;    
    sport_cmd.speed_level() = 0;  

    if (mode == 1) { 
        sport_cmd.body_height() = body_height_target;       
        sport_cmd.foot_raise_height() = foot_raise_height_target; 
    } else if (mode == 0) { 
        // For mode 0 (Idle/Damped), specific height targets are usually ignored.
        // We can set them to 0 explicitly, or rely on the behavior from the declaration's defaults if called with only mode.
        sport_cmd.body_height() = 0.0f; 
        sport_cmd.foot_raise_height() = 0.0f; 
    } else if (mode == 2) { 
        sport_cmd.body_height() = body_height_target; 
        sport_cmd.foot_raise_height() = foot_raise_height_target; 
    } else { 
        sport_cmd.body_height() = body_height_target;
        sport_cmd.foot_raise_height() = foot_raise_height_target;
    }

    sport_cmd.velocity()[0] = 0.0f; 
    sport_cmd.velocity()[1] = 0.0f; 
    sport_cmd.yaw_speed() = 0.0f;   
    sportcmd_publisher->Write(sport_cmd);

    std::cout << "Sent SportModeCmd: mode " << static_cast<int>(sport_cmd.mode())
              << ", gait_type " << static_cast<int>(sport_cmd.gait_type())
              << ", speed_lvl " << static_cast<int>(sport_cmd.speed_level())
              << ", body_h " << std::fixed << std::setprecision(3) << sport_cmd.body_height()
              << ", foot_raise_h " << std::fixed << std::setprecision(3) << sport_cmd.foot_raise_height()
              << ", vel_x " << std::fixed << std::setprecision(3) << sport_cmd.velocity()[0]
              << ", vel_y " << std::fixed << std::setprecision(3) << sport_cmd.velocity()[1]
              << ", yaw_speed " << std::fixed << std::setprecision(3) << sport_cmd.yaw_speed()
              << std::endl;
}