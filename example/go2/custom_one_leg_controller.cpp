#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>    // For M_PI
#include <thread>   // For sleep, this_thread
#include <atomic>

// Unitree SDK Headers
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
// Using B2 MotionSwitcherClient since Go2 version missing
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp> 

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

// Constants for motor commands
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

std::atomic<bool> running(true);

uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }
            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

class CustomOneLegController
{
public:
    CustomOneLegController()
    {
        stand_interpolation_time_ = 0;
    }
    ~CustomOneLegController()
    {
        running = false;
    }

    void Init();
    void StartMotionControl();

private:
    void InitLowCmdDefaults();
    void LowStateMessageHandler(const void* message);
    int queryMotionStatus();
    std::string queryServiceName(std::string form, std::string name);
    void LowCmdWriteStandupCallback();

private:
    float Kp = 60.0f;
    float Kd = 5.0f;
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = 0.002f; // Control period in seconds

    MotionSwitcherClient msc;

    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::LowState_ low_state{};
    unitree_go::msg::dds_::LowState_ current_low_state{};

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    ThreadPtr lowCmdWriteThreadPtr;

    float all_qInit[12] = {0};

    int stand_interpolation_time_ = 0;

    // Multi-step target poses
    float _targetPos_1[12] = {
        0.0f, 1.36f, -2.65f, 0.0f, 1.36f, -2.65f,
        -0.2f, 1.36f, -2.65f, 0.2f, 1.36f, -2.65f
    };

    float _targetPos_2[12] = {
        0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f,
        0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f
    };

    float _targetPos_3[12] = {
        -0.35f, 1.36f, -2.65f, 0.35f, 1.36f, -2.65f,
        -0.5f, 1.36f, -2.65f, 0.5f, 1.36f, -2.65f
    };

    float _startPos[12];
    float _duration_1 = 500;
    float _duration_2 = 500;
    float _duration_3 = 1000;
    float _duration_4 = 900;

    float _percent_1 = 0.0f;
    float _percent_2 = 0.0f;
    float _percent_3 = 0.0f;
    float _percent_4 = 0.0f;

    bool firstRun = true;
    bool done = false;
};

// Implementation

void CustomOneLegController::InitLowCmdDefaults()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF; // Full control level
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++) // Initialize all motors safe stop
    {
        low_cmd.motor_cmd()[i].mode() = 0x01;
        low_cmd.motor_cmd()[i].q() = PosStopF;
        low_cmd.motor_cmd()[i].kp() = 0;
        low_cmd.motor_cmd()[i].dq() = VelStopF;
        low_cmd.motor_cmd()[i].kd() = 0;
        low_cmd.motor_cmd()[i].tau() = 0;
    }
}

void CustomOneLegController::Init()
{
    InitLowCmdDefaults();

    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&CustomOneLegController::LowStateMessageHandler, this, std::placeholders::_1), 1);

    msc.SetTimeout(10.0f);
    msc.Init();

    int max_attempts = 10;
    int attempt_count = 0;
    while (queryMotionStatus() && attempt_count < max_attempts)
    {
        int32_t ret = msc.ReleaseMode();
        if (ret == 0)
        {
            std::cout << "ReleaseMode succeeded (attempt " << attempt_count + 1 << "). Waiting for status." << std::endl;
        }
        else
        {
            std::cout << "ReleaseMode failed (attempt " << attempt_count + 1 << "). Error code: " << ret << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        attempt_count++;
    }

    if (queryMotionStatus() == 0)
    {
        std::cout << "Motion control service deactivated. Ready for low-level control." << std::endl;
    }
    else
    {
        std::cerr << "WARNING: Failed to deactivate motion control after maximum attempts." << std::endl;
    }

    // short delay to receive valid LowState message
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Read current angles from LowState message to initialize the _startPos array
    for (int i = 0; i < 12; ++i)
    {
        _startPos[i] = current_low_state.motor_state()[i].q();
    }

    stand_interpolation_time_ = 0;

    std::cout << "Initial joint positions captured for stand-up interpolation." << std::endl;
}

void CustomOneLegController::StartMotionControl()
{
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writecmd", UT_CPU_ID_NONE, 2000, &CustomOneLegController::LowCmdWriteStandupCallback, this);
}

void CustomOneLegController::LowStateMessageHandler(const void* message)
{
    current_low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

int CustomOneLegController::queryMotionStatus()
{
    std::string robotForm, motionName;
    int32_t ret = msc.CheckMode(robotForm, motionName);
    if (ret != 0)
    {
        std::cerr << "CheckMode failed with error code " << ret << std::endl;
        return 1;
    }
    if (motionName.empty())
        return 0; // Motion control service NOT active (released)
    else
        return 1; // Active
}

std::string CustomOneLegController::queryServiceName(std::string form, std::string name)
{
    if (form == "0")
    {
        if (name == "normal")
            return "sport_mode";
        if (name == "ai")
            return "ai_sport";
        if (name == "advanced")
            return "advanced_sport";
    }
    else
    {
        if (name == "ai-w")
            return "wheeled_sport(go2W)";
        if (name == "normal-w")
            return "wheeled_sport(b2W)";
    }
    return "Unknown Service";
}

void CustomOneLegController::LowCmdWriteStandupCallback()
{
    // On first run capture initial positions from current state
    if (firstRun)
    {
        for (int i = 0; i < 12; ++i)
            _startPos[i] = current_low_state.motor_state()[i].q();

        firstRun = false;
        std::cout << "Captured initial joint positions for stand-up sequence." << std::endl;
    }

    // Linear interpolation helper
    auto lerp = [](float start, float end, float percent) -> float
    {
        if (percent > 1.f)
            percent = 1.f;
        if (percent < 0.f)
            percent = 0.f;
        return start + (end - start) * percent;
    };

    // Send command helper
    auto sendCommand = [&](float* targetPositions)
    {
        for (int i = 0; i < 12; ++i)
        {
            auto& cmd = low_cmd.motor_cmd()[i];
            cmd.mode() = 1;    // Position control
            cmd.q() = targetPositions[i];
            cmd.dq() = 0;
            cmd.tau() = 0;
            cmd.kp() = Kp;
            cmd.kd() = Kd;
        }
        low_cmd.head()[0] = 0xFE;
        low_cmd.head()[1] = 0xEF;
        low_cmd.level_flag() = 0xFF;
        low_cmd.gpio() = 0;
        low_cmd.crc() = crc32_core((uint32_t*)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) / 4) - 1);

        lowcmd_publisher->Write(low_cmd);
    };

    motiontime++;

    if (!done)
    {
        if (motiontime <= _duration_1)
        {
            _percent_1 = motiontime / _duration_1;
            float tempPos[12];
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_startPos[i], _targetPos_1[i], _percent_1);
            sendCommand(tempPos);
        }
        else if (motiontime <= _duration_1 + _duration_2)
        {
            _percent_2 = (motiontime - _duration_1) / _duration_2;
            float tempPos[12];
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_targetPos_1[i], _targetPos_2[i], _percent_2);
            sendCommand(tempPos);
        }
        else if (motiontime <= _duration_1 + _duration_2 + _duration_3)
        {
            _percent_3 = (motiontime - _duration_1 - _duration_2) / _duration_3;
            float tempPos[12];
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_targetPos_2[i], _targetPos_3[i], _percent_3);
            sendCommand(tempPos);
        }
        else
        {
            _percent_4 = (motiontime - _duration_1 - _duration_2 - _duration_3) / _duration_4;
            if (_percent_4 > 1.0f)
                _percent_4 = 1.0f;
            // Hold final pose
            sendCommand(_targetPos_3);
            done = true;
            std::cout << "Stand-up motion complete." << std::endl;
        }
    }
    else
    {
        // Maintain final pose
        sendCommand(_targetPos_3);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <network_interface>" << std::endl;
        return 1;
    }

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WARNING: This program takes DIRECT LOW-LEVEL CONTROL!" << std::endl;
    std::cout << "         Make sure the robot is HUNG UP or LYING ON THE GROUND!" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    try
    {
        ChannelFactory::Instance()->Init(0, std::string(argv[1]));

        CustomOneLegController controller;
        controller.Init();
        controller.StartMotionControl();

        std::cout << "Custom One Leg Controller running in stand-up mode. Press Ctrl+C to exit." << std::endl;

        while (running.load())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred." << std::endl;
        return 1;
    }

    return 0;
}
