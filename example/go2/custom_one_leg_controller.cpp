#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <thread>
#include <atomic>
#include <csignal>

// Unitree SDK Headers
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

std::atomic<bool> running(true);

uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++) {
            CRC32 = (CRC32 & 0x80000000) ? (CRC32 << 1) ^ dwPolynomial : CRC32 << 1;
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

class CustomOneLegController {
public:
    CustomOneLegController() { stand_interpolation_time_ = 0; }
    ~CustomOneLegController() { running = false; }

    void Init();
    void StartMotionControl();
    void RestoreMotionControl();

private:
    void InitLowCmdDefaults();
    void LowStateMessageHandler(const void* message);
    int queryMotionStatus();
    std::string queryServiceName(std::string form, std::string name);
    void LowCmdWriteStandupCallback();

public:
    MotionSwitcherClient msc;

private:
    float Kp = 60.0f, Kd = 5.0f, dt = 0.002f;
    int motiontime = 0;
    bool firstRun = true, done = false;

    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::LowState_ current_low_state{};

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ThreadPtr lowCmdWriteThreadPtr;

    float _startPos[12];
    float _targetPos_1[12] = { 0.0f, 1.36f, -2.65f, 0.0f, 1.36f, -2.65f, -0.2f, 1.36f, -2.65f, 0.2f, 1.36f, -2.65f };
    float _targetPos_2[12] = { 0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f };
    float _targetPos_3[12] = { -0.35f, 1.36f, -2.65f, 0.35f, 1.36f, -2.65f, -0.5f, 1.36f, -2.65f, 0.5f, 1.36f, -2.65f };

    float _duration_1 = 500, _duration_2 = 500, _duration_3 = 1000, _duration_4 = 900;
    float _percent_1 = 0.0f, _percent_2 = 0.0f, _percent_3 = 0.0f, _percent_4 = 0.0f;

    int stand_interpolation_time_ = 0;

    std::string initial_robot_form;
    std::string initial_motion_name;
};

void CustomOneLegController::InitLowCmdDefaults()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++) {
        auto& cmd = low_cmd.motor_cmd()[i];
        cmd.mode() = 0x01;
        cmd.q() = PosStopF;
        cmd.kp() = 0;
        cmd.dq() = VelStopF;
        cmd.kd() = 0;
        cmd.tau() = 0;
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

    // Save initial mode
    msc.CheckMode(initial_robot_form, initial_motion_name);

    int attempt_count = 0;
    const int max_attempts = 10;

    while (queryMotionStatus() && attempt_count < max_attempts) {
        int32_t ret = msc.ReleaseMode();
        if (ret == 0)
            std::cout << "ReleaseMode succeeded (attempt " << attempt_count + 1 << ")" << std::endl;
        else
            std::cout << "ReleaseMode failed (attempt " << attempt_count + 1 << "). Error: " << ret << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++attempt_count;
    }

    if (queryMotionStatus() == 0)
        std::cout << "Motion control deactivated. Ready for low-level control." << std::endl;
    else
        std::cerr << "WARNING: Failed to deactivate motion control after max attempts." << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    for (int i = 0; i < 12; ++i)
        _startPos[i] = current_low_state.motor_state()[i].q();

    std::cout << "Initial joint positions captured." << std::endl;
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
    std::string form, name;
    int ret = msc.CheckMode(form, name);
    return (ret != 0 || !name.empty()) ? 1 : 0;
}

std::string CustomOneLegController::queryServiceName(std::string form, std::string name)
{
    if (form == "0") {
        if (name == "normal") return "sport_mode";
        if (name == "ai") return "ai_sport";
        if (name == "advanced") return "advanced_sport";
    } else {
        if (name == "ai-w") return "wheeled_sport(go2W)";
        if (name == "normal-w") return "wheeled_sport(b2W)";
    }
    return "Unknown Service";
}

void CustomOneLegController::RestoreMotionControl()
{
    if (!initial_motion_name.empty()) {
        int32_t ret = msc.SelectMode(initial_motion_name);
        if (ret == 0) {
            std::cout << "Successfully restored motion control mode: " 
                      << queryServiceName(initial_robot_form, initial_motion_name) << std::endl;
        } else {
            std::cerr << "Failed to restore mode. Error: " << ret << std::endl;
        }
    } else {
        std::cerr << "Initial mode unknown. Cannot restore." << std::endl;
    }
}

void CustomOneLegController::LowCmdWriteStandupCallback()
{
    if (firstRun) {
        for (int i = 0; i < 12; ++i)
            _startPos[i] = current_low_state.motor_state()[i].q();
        firstRun = false;
        std::cout << "Captured initial joint positions." << std::endl;
    }

    auto lerp = [](float a, float b, float p) {
        return a + std::clamp(p, 0.0f, 1.0f) * (b - a);
    };

    auto sendCommand = [&](float* target) {
        for (int i = 0; i < 12; ++i) {
            auto& cmd = low_cmd.motor_cmd()[i];
            cmd.mode() = 1;
            cmd.q() = target[i];
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

    if (!done) {
        float tempPos[12];

        if (motiontime <= _duration_1) {
            _percent_1 = motiontime / _duration_1;
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_startPos[i], _targetPos_1[i], _percent_1);
        } else if (motiontime <= _duration_1 + _duration_2) {
            _percent_2 = (motiontime - _duration_1) / _duration_2;
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_targetPos_1[i], _targetPos_2[i], _percent_2);
        } else if (motiontime <= _duration_1 + _duration_2 + _duration_3) {
            _percent_3 = (motiontime - _duration_1 - _duration_2) / _duration_3;
            for (int i = 0; i < 12; ++i)
                tempPos[i] = lerp(_targetPos_2[i], _targetPos_3[i], _percent_3);
        } else {
            done = true;
            std::cout << "Stand-up motion complete." << std::endl;
            for (int i = 0; i < 12; ++i)
                tempPos[i] = _targetPos_3[i];
        }

        sendCommand(tempPos);
    } else {
        sendCommand(_targetPos_3); // Hold pose
    }
}

std::unique_ptr<CustomOneLegController> controller;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <network_interface>" << std::endl;
        return 1;
    }

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WARNING: This program takes DIRECT LOW-LEVEL CONTROL!" << std::endl;
    std::cout << "         Make sure the robot is HUNG UP or LYING ON THE GROUND!" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    std::signal(SIGINT, [](int signal) {
        std::cout << "\nCtrl+C received. Shutting down..." << std::endl;
        running = false;
        if (controller) {
            controller->RestoreMotionControl();
        }
    });

    try {
        ChannelFactory::Instance()->Init(0, std::string(argv[1]));

        controller = std::make_unique<CustomOneLegController>();
        controller->Init();
        controller->StartMotionControl();

        std::cout << "Custom One Leg Controller running. Press Ctrl+C to exit." << std::endl;

        while (running.load())
            std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
        return 1;
    }

    return 0;
}
