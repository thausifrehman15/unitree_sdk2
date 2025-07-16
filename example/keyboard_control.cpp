#include <iostream>
#include <unistd.h>     // For usleep, read, sleep
#include <termios.h>    // For non-blocking keyboard input
#include <fcntl.h>      // For fcntl
#include <signal.h>     // For signal handling (Ctrl+C)
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <atomic>

// Unitree SDK Headers
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include "unitree/common/thread/thread.hpp"
#include "unitree/common/time/time_tool.hpp"

// Include the IDL header for sensor (LowState) messages and low-level commands.
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"

// Topic definition for sensor data and lowcmd
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_LOWCMD   "rt/lowcmd"

// Define movement speeds (adjust as needed)
const float MOVE_SPEED_X = 0.3f;   // m/s
const float MOVE_SPEED_Y = 0.3f;   // m/s
const float ROTATE_SPEED   = 0.5f; // rad/s

// Define the lowcmd publisher (global definition)
std::shared_ptr<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>> lowcmd_publisher;

// Global variable to store the latest LowState data received
unitree_go::msg::dds_::LowState_ current_low_state_kc{};
// Flag to indicate if we have received the first LowState message
std::atomic_bool state_received_kc(false);
// Global pointer for the LowState subscriber
std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>> lowstate_subscriber_kc;


// Global variable to control the main loop
volatile bool running = true;

// Global SportClient pointer to be accessible by signal handler
unitree::robot::go2::SportClient* sport_client_ptr = nullptr;

// Terminal settings variables
struct termios old_tio, new_tio;

//New Callback function
void LowStateMessageHandlerKC(const void* msg) {
    current_low_state_kc = *(static_cast<const unitree_go::msg::dds_::LowState_*>(msg));
    state_received_kc.store(true); // Set the flag to true when the first message is received
}

// Function to set terminal to non-blocking mode
void set_terminal_mode() {
    tcgetattr(STDIN_FILENO, &old_tio);  // Save current terminal settings
    new_tio = old_tio;
    new_tio.c_lflag &= (~ICANON & ~ECHO);  // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);  // Apply new settings
    // Set stdin to non-blocking
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
}

// Function to restore terminal settings
void restore_terminal_mode() {
    tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);  // Restore old settings
    // Optionally restore blocking mode:
    // fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) & ~O_NONBLOCK);
}

// Signal handler for Ctrl+C (SIGINT)
void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Stopping robot and exiting." << std::endl;
    if (sport_client_ptr) {
        // Command the robot to stop moving immediately
        sport_client_ptr->StopMove();
        usleep(500000);  // Give half a second to process
        // Optionally, command it to sit down or crouch:
        // sport_client_ptr->StandDown();
    }
    running = false;  // Signal the main loop to terminate
}

// A helper function to compute CRC for the command message.
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
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

// This function commands the robot to sit on two legs.
// The hind legs are tuned to a "sitting" (bent) posture while the front legs remain in a neutral (idle) configuration.
void SitOnTwoLegs() {
    // Create a LowCmd message.
    unitree_go::msg::dds_::LowCmd_ low_cmd;
    
    // Set header and default parameters.
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;
    
    // Define target positions for 12 motors (adjusted for realistic movement).
    // For front-left leg (motors 0, 1, 2):
    // - Move the leg into a position where it bends, similar to a hand move.
    float front_left_leg_target[3] = {1.0f, 1.5f, -3.5f}; // Example values for hand movement.
    
    // Define neutral position for all other motors (other legs remain neutral for now).
    float neutral_leg_target[6] = {0.0f, 1.36f, -2.65f, 0.0f, 1.36f, -2.65f};
    
    // Configure motor commands for the 12 motors.
    for (int i = 0; i < 12; i++) {
        low_cmd.motor_cmd()[i].mode() = 0x01;  // Set to servo mode.
        
        if (i == 0 || i == 1 || i == 2) {
            // Move the front-left leg (motor 0, 1, 2).
            low_cmd.motor_cmd()[i].q() = front_left_leg_target[i];
        } else {
            // Keep other motors (legs) in neutral position.
            low_cmd.motor_cmd()[i].q() = neutral_leg_target[i % 6];
        }
        
        // Set control gains and other parameters.
        low_cmd.motor_cmd()[i].kp() = 60;
        low_cmd.motor_cmd()[i].dq() = 16000.0f;
        low_cmd.motor_cmd()[i].kd() = 5;
        low_cmd.motor_cmd()[i].tau() = 0;
    }
    
    // Compute and set the CRC for the message.
    low_cmd.crc() = crc32_core((uint32_t*)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    
    // Publish the command.
    lowcmd_publisher->Write(low_cmd);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_interface_name>" << std::endl;
        return 1; // Exit with an error code
    }
    // Initialize the channel factory with domain ID 0 and the network interface name from argv[1]
    // argv[1] is a char*, convert it to std::string
    unitree::robot::ChannelFactory::Instance()->Init(0, std::string(argv[1]));
    // Initialize lowcmd publisher.
    lowcmd_publisher = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(TOPIC_LOWCMD);
    lowcmd_publisher->InitChannel();  // Initialize the channel as required.
    
    // Initialize sensor subscriber for low state messages 
    lowstate_subscriber_kc = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(TOPIC_LOWSTATE);
    lowstate_subscriber_kc->InitChannel(LowStateMessageHandlerKC, 1);
    
    // Create and initialize SportClient.
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f); // Set communication timeout.
    sport_client.Init();            // Initialize the client.
    sport_client_ptr = &sport_client;  // Assign global pointer for signal handler.

    // Setup signal handler for graceful exit on Ctrl+C.
    signal(SIGINT, sigint_handler);

    // Set terminal to non-blocking mode.
    set_terminal_mode();

    // Variables to hold current movement commands.
    float target_vx = 0.0f;     // Forward/backward velocity (m/s)
    float target_vy = 0.0f;     // Left/right velocity (m/s)
    float target_vyaw = 0.0f;   // Rotational velocity (rad/s)
    bool walking = false;       // Track if a walk command should be active

    // Print available commands.
    std::cout << "Keyboard Control Initialized." << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << " W: Forward | S: Backward        " << std::endl;
    std::cout << " A: Left    | D: Right           " << std::endl;
    std::cout << " Q: Turn L  | E: Turn R          " << std::endl;
    std::cout << " U: Stand Up| J: Sit Down        " << std::endl;
    std::cout << " Spacebar: Stop Motion           " << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Additional controls:" << std::endl;
    std::cout << " 1: Damp          2: BalanceStand" << std::endl;
    std::cout << " 3: RecoveryStand 4: Euler (0,0,0)" << std::endl;
    std::cout << " 5: Sit           6: RiseSit" << std::endl;
    // std::cout << " 7: SwitchGait    8: Trigger" << std::endl; // Commented out
    // std::cout << " 9: BodyHeight    0: FootRaiseHeight" << std::endl; // Commented out
    std::cout << " -: SpeedLevel" << std::endl;
    std::cout << " H: Hello         X: Stretch" << std::endl;
    // std::cout << " C: TrajectoryFollow" << std::endl; // Commented out
    std::cout << " V: SwitchJoystick  " /*B: ContinuousGait"*/ << std::endl; // ContinuousGait commented out
    // std::cout << " N: Wallow        M: Content" << std::endl; // Wallow commented out
    std::cout << " M: Content" << std::endl; // Re-arranged print
    std::cout << " R: Heart         P: Pose" << std::endl;
    std::cout << " K: Scrape (with sensor log)" << std::endl;
    std::cout << " F: FrontFlip     O: FrontJump" << std::endl;
    std::cout << " I: FrontPounce   G: Dance1" << std::endl;
    std::cout << " L: Dance2        " /*T: Dance3"*/ << std::endl; // Dance3 commented out
    // std::cout << " Y: Dance4        Z: HopSpinLeft" << std::endl; // Dance4, HopSpinLeft commented out
    // std::cout << " !: HopSpinRight  @: WiggleHips" << std::endl; // HopSpinRight, WiggleHips commented out
    // std::cout << " #: GetState      $: EconomicGait" << std::endl; // GetState commented out, EconomicGait will be fixed
    std::cout << " $: EconomicGait" << std::endl; // Re-arranged print
    std::cout << " %: LeftFlip      ^: BackFlip" << std::endl;
    std::cout << " &: FreeWalk      *: FreeBound" << std::endl;
    std::cout << " (: FreeJump      ): FreeAvoid" << std::endl;
    // std::cout << "_: WalkStair     +: WalkUpright" << std::endl; // WalkStair commented out
    std::cout << "+: WalkUpright" << std::endl; // Re-arranged print
    std::cout << " ;: SitOnTwoLegs  (custom movement)" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Robot is currently crouched. Press 'U' to stand." << std::endl;

    // Main control loop.
    while (running) {
        char c = 0;
        int nread = read(STDIN_FILENO, &c, 1); // Read one character non-blockingly.

        if (nread > 0) {
            switch (c) {
                // --- Movement and Posture Commands ---
                case 'u':
                case 'U':
                    std::cout << "Command: Stand Up" << std::endl;
                    sport_client.StandUp();
                    target_vx = target_vy = target_vyaw = 0.0f;
                    walking = false;
                    break;
                case 'j':
                case 'J':
                    std::cout << "Command: Sit Down" << std::endl;
                    sport_client.StandDown();
                    target_vx = target_vy = target_vyaw = 0.0f;
                    walking = false;
                    break;
                case 'w':
                case 'W':
                    std::cout << "Command: Forward" << std::endl;
                    target_vx = MOVE_SPEED_X; target_vy = 0.0f; target_vyaw = 0.0f;
                    walking = true;
                    break;
                case 's':
                case 'S':
                    std::cout << "Command: Backward" << std::endl;
                    target_vx = -MOVE_SPEED_X; target_vy = 0.0f; target_vyaw = 0.0f;
                    walking = true;
                    break;
                case 'a':
                case 'A':
                    std::cout << "Command: Strafe Left" << std::endl;
                    target_vx = 0.0f; target_vy = MOVE_SPEED_Y; target_vyaw = 0.0f;
                    walking = true;
                    break;
                case 'd':
                case 'D':
                    std::cout << "Command: Strafe Right" << std::endl;
                    target_vx = 0.0f; target_vy = -MOVE_SPEED_Y; target_vyaw = 0.0f;
                    walking = true;
                    break;
                case 'q':
                case 'Q':
                    std::cout << "Command: Turn Left" << std::endl;
                    target_vx = 0.0f; target_vy = 0.0f; target_vyaw = ROTATE_SPEED;
                    walking = true;
                    break;
                case 'e':
                case 'E':
                    std::cout << "Command: Turn Right" << std::endl;
                    target_vx = 0.0f; target_vy = 0.0f; target_vyaw = -ROTATE_SPEED;
                    walking = true;
                    break;
                case ' ':
                    std::cout << "Command: Stop Motion" << std::endl;
                    target_vx = target_vy = target_vyaw = 0.0f;
                    walking = false;
                    sport_client.StopMove();
                    break;

                // --- Additional Mappings (Numeric & Punctuation) ---
                case '1':
                    std::cout << "Command: Damp" << std::endl;
                    sport_client.Damp();
                    break;
                case '2':
                    std::cout << "Command: BalanceStand" << std::endl;
                    sport_client.BalanceStand();
                    break;
                case '3':
                    std::cout << "Command: RecoveryStand" << std::endl;
                    sport_client.RecoveryStand();
                    break;
                case '4':
                    std::cout << "Command: Euler (0,0,0)" << std::endl;
                    sport_client.Euler(0.0f, 0.0f, 0.0f);
                    break;
                case '5': // Key for High-Level Sit
                    std::cout << "Command: Sit (High Level)" << std::endl;

                    // --- Log Initial Positions FIRST ---
                    // Check if the subscriber has received data
                    if (state_received_kc.load()) {
                        std::cout << "--- Initial Joint Positions (Before Sit Command) ---" << std::endl;
                        // Loop through all 12 motors
                        for (int i = 0; i < 12; ++i) {
                            float initial_q = current_low_state_kc.motor_state()[i].q();
                            std::cout << "Joint " << i << ": " << initial_q << " rad" << std::endl;
                        }
                        std::cout << "-------------------------------------------------" << std::endl;
                    } else {
                        std::cout << "Warning: LowState not received yet. Cannot log initial positions." << std::endl;
                    }
                    // --- End Log Initial Positions ---


                    sport_client.Sit(); // Send the high-level command

                    // Wait for the robot to physically execute the command
                    std::cout << "Waiting 2 seconds for robot to settle into Sit pose..." << std::endl;
                    sleep(2); // Pause execution for 2 seconds (adjust if needed)

                    // --- Read and Print Joint Angles ---
                    if (state_received_kc.load())
                    {
                        std::cout << "--- Current Joint Positions (After High-Level Sit) ---" << std::endl;
                        // Loop through all 12 motors
                        for (int i = 0; i < 12; ++i) {
                            float current_q = current_low_state_kc.motor_state()[i].q();
                            std::cout << "Joint " << i << ": " << current_q << " rad" << std::endl;
                        }
                        std::cout << "------------------------------------------------------" << std::endl;
                        std::cout << "==> Use these 12 'q' values as targets (front_neutral and hind_sit)" << std::endl;
                        std::cout << "    in your sitonhindlegs.cpp code! <==" << std::endl;
                    } else {
                        // Error message if the subscriber hasn't received data
                        std::cout << "Error: LowState data has not been received yet." << std::endl;
                        std::cout << "       Make sure the LowState subscriber is correctly initialized earlier in main()." << std::endl;
                    }

                    // Reset movement state variables for keyboard_control
                    target_vx = target_vy = target_vyaw = 0.0f;
                    walking = false;
                    break;
                case '6':
                    std::cout << "Command: RiseSit" << std::endl;
                    sport_client.RiseSit();
                    break;
                /*
                case '7':
                    std::cout << "Command: SwitchGait(1)" << std::endl;
                    sport_client.SwitchGait(1); // Error: No member named 'SwitchGait'
                    break;
                case '8':
                    std::cout << "Command: Trigger" << std::endl;
                    sport_client.Trigger(); // Error: No member named 'Trigger'
                    break;
                case '9':
                    std::cout << "Command: BodyHeight(0.5)" << std::endl;
                    sport_client.BodyHeight(0.5f); // Error: No member named 'BodyHeight'
                    break;
                case '0':
                    std::cout << "Command: FootRaiseHeight(0.1)" << std::endl;
                    sport_client.FootRaiseHeight(0.1f); // Error: No member named 'FootRaiseHeight'
                    break;
                */
                case '-':
                    std::cout << "Command: SpeedLevel(1)" << std::endl;
                    sport_client.SpeedLevel(1);
                    break;

                // --- Additional Mappings (Alphabet & Special Characters) ---
                case 'h':
                case 'H':
                    std::cout << "Command: Hello" << std::endl;
                    sport_client.Hello();
                    break;
                case 'x':
                case 'X':
                    std::cout << "Command: Stretch" << std::endl;
                    sport_client.Stretch();
                    break;
                /*
                case 'c':
                case 'C': {
                    std::cout << "Command: TrajectoryFollow" << std::endl;
                    std::vector<unitree::robot::go2::PathPoint> path; // Empty/default path.
                    sport_client.TrajectoryFollow(path); // Error: No member named 'TrajectoryFollow'
                    break;
                }
                */
                case 'v':
                case 'V':
                    std::cout << "Command: SwitchJoystick(true)" << std::endl;
                    sport_client.SwitchJoystick(true);
                    break;
                /*
                case 'b':
                case 'B':
                    std::cout << "Command: ContinuousGait(true)" << std::endl;
                    sport_client.ContinuousGait(true); // Error: No member named 'ContinuousGait'
                    break;
                case 'n':
                case 'N':
                    std::cout << "Command: Wallow" << std::endl;
                    sport_client.Wallow(); // Error: No member named 'Wallow'
                    break;
                */
                case 'm':
                case 'M':
                    std::cout << "Command: Content" << std::endl;
                    sport_client.Content();
                    break;
                case 'r':
                case 'R':
                    std::cout << "Command: Heart" << std::endl;
                    sport_client.Heart();
                    break;
                case 'p':
                case 'P':
                    std::cout << "Command: Pose(true)" << std::endl;
                    sport_client.Pose(true);
                    break;
                case 'k':
                case 'K': {
                    std::cout << "Command: Scrape" << std::endl;
                    sport_client.Scrape();
                    break;
                }
                case 'f':
                case 'F':
                    std::cout << "Command: FrontFlip" << std::endl;
                    sport_client.FrontFlip();
                    break;
                case 'o':
                case 'O':
                    std::cout << "Command: FrontJump" << std::endl;
                    sport_client.FrontJump();
                    break;
                case 'i':
                case 'I':
                    std::cout << "Command: FrontPounce" << std::endl;
                    sport_client.FrontPounce();
                    break;
                case 'g':
                case 'G':
                    std::cout << "Command: Dance1" << std::endl;
                    sport_client.Dance1();
                    break;
                case 'l':
                case 'L':
                    std::cout << "Command: Dance2" << std::endl;
                    sport_client.Dance2();
                    break;
                case 't':
                case 'T':
                    std::cout << "Command: Dance3 (changed to Dance1)" << std::endl;
                    sport_client.Dance1(); // Compiler suggested Dance1
                    break;
                case 'y':
                case 'Y':
                    std::cout << "Command: Dance4 (changed to Dance1)" << std::endl;
                    sport_client.Dance1(); // Compiler suggested Dance1
                    break;
                /*
                case 'z':
                case 'Z':
                    std::cout << "Command: HopSpinLeft" << std::endl;
                    sport_client.HopSpinLeft(); // Error: No member named 'HopSpinLeft'
                    break;
                case '!':
                    std::cout << "Command: HopSpinRight" << std::endl;
                    sport_client.HopSpinRight(); // Error: No member named 'HopSpinRight'
                    break;
                case '@':
                    std::cout << "Command: WiggleHips" << std::endl;
                    sport_client.WiggleHips(); // Error: No member named 'WiggleHips'
                    break;
                case '#': {
                    std::cout << "Command: GetState" << std::endl;
                    std::vector<std::string> keys;
                    std::map<std::string, std::string> state;
                    sport_client.GetState(keys, state); // Error: No member named 'GetState'
                    break;
                }
                */
                case '$':
                    std::cout << "Command: EconomicGait()" << std::endl;
                    sport_client.EconomicGait(); // Corrected: Removed argument based on compiler error
                    break;
                case '%':
                    std::cout << "Command: LeftFlip" << std::endl;
                    sport_client.LeftFlip();
                    break;
                case '^':
                    std::cout << "Command: BackFlip" << std::endl;
                    sport_client.BackFlip();
                    break;
                case '&':
                    std::cout << "Command: FreeWalk" << std::endl;
                    sport_client.FreeWalk();
                    break;
                case '*':
                    std::cout << "Command: FreeBound(true)" << std::endl;
                    sport_client.FreeBound(true);
                    break;
                case '(':
                    std::cout << "Command: FreeJump(true)" << std::endl;
                    sport_client.FreeJump(true);
                    break;
                case ')':
                    std::cout << "Command: FreeAvoid(true)" << std::endl;
                    sport_client.FreeAvoid(true);
                    break;
                /*
                case '_':
                    std::cout << "Command: WalkStair(true)" << std::endl;
                    sport_client.WalkStair(true); // Error: No member named 'WalkStair'
                    break;
                */
                case '+':
                    std::cout << "Command: WalkUpright(true)" << std::endl;
                    sport_client.WalkUpright(true);
                    break;
                    
                // --- Custom Mapping for Two-Legged Sit ---
                case ';':
                    std::cout << "Command: SitOnTwoLegs" << std::endl;
                    SitOnTwoLegs();
                    break;
                    
                default:
                    // Optionally handle unassigned keys.
                    break;
            }
        }
        
        // If a movement command is active, continuously send the Move command.
        if (walking) {
            sport_client.Move(target_vx, target_vy, target_vyaw);
        }
        
        // Small delay to prevent busy waiting (20 ms = 50 Hz loop).
        usleep(20000);
    }

    // Cleanup before exiting.
    std::cout << "Restoring terminal settings..." << std::endl;
    restore_terminal_mode();
    std::cout << "Exiting program." << std::endl;

    return 0;
}