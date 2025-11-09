#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <signal.h>

// Unitree SDK Headers
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include "unitree/robot/go2/video/video_client.hpp"
#include "unitree/common/thread/thread.hpp"
#include "unitree/common/time/time_tool.hpp"

// OpenCV Headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <alsa/asoundlib.h>
#include "whisper.h"  // whisper.h

// Global variable to control the main loop
std::atomic<bool> running(true);
// Global SportClient pointer accessible by signal handler
unitree::robot::go2::SportClient* sport_client_ptr = nullptr;

// --- Robot Control Constants ---
const float MIN_FOLLOW_SPEED_X = 0.5f;
const float MAX_FOLLOW_SPEED_X = 0.8f;
const float TURN_SPEED_YAW = 0.8f;

const float PERSON_VERY_CLOSE_HEIGHT_RATIO = 0.95f;
const float PERSON_CLOSE_HEIGHT_RATIO      = 0.65f;
const float PERSON_MEDIUM_HEIGHT_RATIO     = 0.45f;
const float PERSON_FAR_HEIGHT_RATIO        = 0.25f;
const float PERSON_LOST_HEIGHT_RATIO       = 0.15f;

const float RESUME_FOLLOW_HEIGHT_RATIO     = 0.4f;
const float HORIZONTAL_CENTER_TOLERANCE    = 0.15f;

const int FRAME_RATE = 100;
const int SIT_ON_CLOSE_DELAY_FRAMES = 5 * FRAME_RATE;
const int SIT_ON_STILL_DELAY_FRAMES = 3 * FRAME_RATE;
const int LOST_PERSON_SIT_DELAY_FRAMES = 1 * FRAME_RATE;
const int ACTION_WAIT_MS = 1500;

// Manual control constants
const float MANUAL_MOVE_SPEED = 0.3f;
const float MANUAL_TURN_SPEED = 0.7f;

// Audio constants (voice only, no clap)
static unsigned int AUDIO_SAMPLE_RATE = 16000;
static unsigned int AUDIO_SAMPLE_RATE_FOR_WHISPER = 16000;
const int AUDIO_CHANNELS = 1;
const int AUDIO_FRAMES = 4096;

std::atomic<bool> speech_command_ready(false);
std::string last_audio_command = "";
std::mutex command_mutex;
std::atomic<bool> is_spinning(false);
std::atomic<bool> push_to_talk_active(false);
const float TURN_AROUND_SPEED = 1.0f;
const float CONTINUOUS_SPIN_SPEED = 0.8f;

// --- Robot State Enum ---
enum RobotState {
    IDLE_STANDING,
    SITTING,
    STANDING_UP_IN_PROGRESS,
    SITTING_DOWN_IN_PROGRESS,
    FOLLOWING,
    STOPPING_FOR_STILL_PERSON,
    LOST_PERSON_IDLE,
    SPINNING
};

// Voice command keywords
const std::string WAKE_WORD = "robot";
const std::vector<std::string> WAKE_WORD_ALIASES = {
    "robot", "robert", "robots", "bought", "bot"
};
const std::vector<std::string> WAKE_WORDS = {"follow", "follow me", "come here", "start"};
const std::vector<std::string> STOP_WORDS = {"stop", "wait", "stay", "halt"};
const std::vector<std::string> SIT_WORDS = {"sit", "sit down", "rest", "sleep"};
const std::vector<std::string> STAND_WORDS = {"stand", "stand up", "get up", "wake up"};
const std::vector<std::string> TURN_AROUND_WORDS = {"turn around", "turn back", "about face", "reverse"};
const std::vector<std::string> SPIN_WORDS = {"spin", "rotate", "keep turning", "keep spinning"};

const std::vector<std::string> DAMP_WORDS = {"damp", "dampen", "relax"};
const std::vector<std::string> BALANCE_WORDS = {"balance", "balance stand"};
const std::vector<std::string> RECOVERY_WORDS = {"recovery", "recover", "recovery stand"};
const std::vector<std::string> HELLO_WORDS = {"hello", "wave", "greet"};
const std::vector<std::string> STRETCH_WORDS = {"stretch", "stretch out"};
const std::vector<std::string> DANCE_WORDS = {"dance", "dance one", "dance two"};
const std::vector<std::string> FLIP_WORDS = {"flip", "front flip", "back flip"};
const std::vector<std::string> JUMP_WORDS = {"jump", "front jump"};

// Audio detection class using Whisper
class AudioDetector {
public:
    AudioDetector() : running(false), whisper_ctx(nullptr), capture_handle(nullptr) {
        // Initialize Whisper model with new API
        std::string model_path = "/home/slam22/unitree_ws/src/unitree_sdk2/assets/whisper.cpp/models/ggml-tiny.en.bin";

        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false;  // Set to true if you have GPU support

        whisper_ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);

        if (!whisper_ctx) {
            std::cerr << "WARNING: Could not load Whisper model from " << model_path << std::endl;
            std::cerr << "   Speech recognition disabled." << std::endl;
            std::cerr << "   Run: cd ~/unitree_ws/src/unitree_sdk2/assets/whisper.cpp && bash ./models/download-ggml-model.sh tiny.en" << std::endl;
        } else {
            std::cout << "Whisper speech recognition initialized! (Model: tiny.en)" << std::endl;
        }
    }

    ~AudioDetector() {
        running.store(false);
        if (whisper_ctx) whisper_free(whisper_ctx);
    }

    void start() {
        if (running.load() || !whisper_ctx) return;
        running.store(true);
        audio_thread = std::thread(&AudioDetector::audioLoop, this);
    }

    void stop() {
        if (!running.load()) return;

        std::cout << "Stopping audio detector..." << std::endl;
        running.store(false);

        if (capture_handle != nullptr) {
            std::cout << "   Closing audio device..." << std::endl;
            snd_pcm_drop(capture_handle);
            snd_pcm_close(capture_handle);
            capture_handle = nullptr;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        if (audio_thread.joinable()) {
            audio_thread.detach();
        }

        std::cout << "Audio detector stopped" << std::endl;
    }

    bool isSpeechCommandReady() {
        if (!push_to_talk_active.load()) return false;

        bool result = speech_command_ready.load();
        if (result) {
            speech_command_ready.store(false);
            return true;
        }
        return false;
    }

    std::string getLastCommand() {
        std::lock_guard<std::mutex> lock(command_mutex);
        std::string cmd = last_audio_command;
        last_audio_command = "";
        return cmd;
    }

private:
    std::atomic<bool> running;
    std::thread audio_thread;
    whisper_context* whisper_ctx;
    snd_pcm_t* capture_handle;

    static constexpr int WHISPER_SAMPLE_RATE_CUSTOM = 16000;
    static constexpr int CHUNK_DURATION_MS = 3000;
    static constexpr int CHUNK_SAMPLES = WHISPER_SAMPLE_RATE_CUSTOM * CHUNK_DURATION_MS / 1000;

    bool containsKeyword(const std::string& text, const std::vector<std::string>& keywords) {
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

        for (const auto& keyword : keywords) {
            if (lower_text.find(keyword) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool containsWakeWordAlias(const std::string& text) {
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

        for (const auto& alias : WAKE_WORD_ALIASES) {
            if (lower_text.find(alias) != std::string::npos) {
                if (alias != "robot") {
                    std::cout << "   Accepted alias: '" << alias << "' -> 'robot'" << std::endl;
                }
                return true;
            }
        }
        return false;
    }

    void processCommand(const std::string& text) {
        if (text.empty() || text.length() < 3) return;
        if (!push_to_talk_active.load()) {
            return;
        }

        std::cout << "Recognized: \"" << text << "\"" << std::endl;

        // Check for wake word
        if (!containsWakeWordAlias(text)) {
            std::cout << "   No wake word detected. Ignoring." << std::endl;
            return;
        }

        std::cout << "   Wake word detected! Processing command..." << std::endl;

        std::lock_guard<std::mutex> lock(command_mutex);

        // Check for commands (order matters)
        if (containsKeyword(text, TURN_AROUND_WORDS)) {
            last_audio_command = "turn_around";
            speech_command_ready.store(true);
            std::cout << "Voice Command: TURN AROUND 180" << std::endl;
        } else if (containsKeyword(text, SPIN_WORDS)) {
            last_audio_command = "spin";
            speech_command_ready.store(true);
            std::cout << "Voice Command: CONTINUOUS SPIN" << std::endl;
        } else if (containsKeyword(text, WAKE_WORDS)) {
            last_audio_command = "follow";
            speech_command_ready.store(true);
            std::cout << "Voice Command: FOLLOW ME" << std::endl;
        } else if (containsKeyword(text, STOP_WORDS)) {
            last_audio_command = "stop";
            speech_command_ready.store(true);
            std::cout << "Voice Command: STOP" << std::endl;
        } else if (containsKeyword(text, SIT_WORDS)) {
            last_audio_command = "sit";
            speech_command_ready.store(true);
            std::cout << "Voice Command: SIT DOWN" << std::endl;
        } else if (containsKeyword(text, STAND_WORDS)) {
            last_audio_command = "stand";
            speech_command_ready.store(true);
            std::cout << "Voice Command: STAND UP" << std::endl;
        } else if (containsKeyword(text, DAMP_WORDS)) {
            last_audio_command = "damp";
            speech_command_ready.store(true);
            std::cout << "Voice Command: DAMP" << std::endl;
        } else if (containsKeyword(text, BALANCE_WORDS)) {
            last_audio_command = "balance";
            speech_command_ready.store(true);
            std::cout << "Voice Command: BALANCE STAND" << std::endl;
        } else if (containsKeyword(text, RECOVERY_WORDS)) {
            last_audio_command = "recovery";
            speech_command_ready.store(true);
            std::cout << "Voice Command: RECOVERY STAND" << std::endl;
        } else if (containsKeyword(text, HELLO_WORDS)) {
            last_audio_command = "hello";
            speech_command_ready.store(true);
            std::cout << "Voice Command: HELLO" << std::endl;
        } else if (containsKeyword(text, STRETCH_WORDS)) {
            last_audio_command = "stretch";
            speech_command_ready.store(true);
            std::cout << "Voice Command: STRETCH" << std::endl;
        } else if (containsKeyword(text, DANCE_WORDS)) {
            last_audio_command = "dance";
            speech_command_ready.store(true);
            std::cout << "Voice Command: DANCE" << std::endl;
        } else if (containsKeyword(text, FLIP_WORDS)) {
            last_audio_command = "flip";
            speech_command_ready.store(true);
            std::cout << "Voice Command: FLIP" << std::endl;
        } else if (containsKeyword(text, JUMP_WORDS)) {
            last_audio_command = "jump";
            speech_command_ready.store(true);
            std::cout << "Voice Command: JUMP" << std::endl;
        }
    }

    void audioLoop() {
        snd_pcm_hw_params_t* hw_params;
        int err;

        capture_handle = nullptr;

        if ((err = snd_pcm_open(&capture_handle, "default", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
            std::cerr << "Cannot open audio device: " << snd_strerror(err) << std::endl;
            capture_handle = nullptr;
            return;
        }

        snd_pcm_hw_params_malloc(&hw_params);
        snd_pcm_hw_params_any(capture_handle, hw_params);
        snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE);

        unsigned int sample_rate = WHISPER_SAMPLE_RATE_CUSTOM;
        snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &sample_rate, 0);
        snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1);

        if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
            std::cerr << "Cannot set parameters: " << snd_strerror(err) << std::endl;
            snd_pcm_close(capture_handle);
            return;
        }

        snd_pcm_hw_params_free(hw_params);
        snd_pcm_prepare(capture_handle);
        snd_pcm_nonblock(capture_handle, 1);

        std::vector<float> audio_buffer;
        std::vector<int16_t> pcm_buffer(4096);

        std::cout << "Whisper recognition started (rate: " << sample_rate << " Hz)" << std::endl;
        std::cout << "Say 'robot' followed by a command (follow, stop, sit, dance, etc.)" << std::endl;

        while (running.load()) {
            err = snd_pcm_readi(capture_handle, pcm_buffer.data(), pcm_buffer.size());

            if (err == -EAGAIN) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (err < 0) {
                if (err == -EPIPE) {
                    snd_pcm_prepare(capture_handle);
                }
                continue;
            }

            // Convert int16 to float [-1.0, 1.0]
            for (int i = 0; i < err; i++) {
                audio_buffer.push_back(static_cast<float>(pcm_buffer[i]) / 32768.0f);
            }

            // Process when we have enough audio (3 seconds)
            if (audio_buffer.size() >= CHUNK_SAMPLES) {
                // Setup Whisper parameters
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wparams.language = "en";
                wparams.translate = false;
                wparams.no_timestamps = true;
                wparams.single_segment = true;
                wparams.print_realtime = false;
                wparams.print_progress = false;
                wparams.print_timestamps = false;
                wparams.print_special = false;

                // Run inference
                if (whisper_full(whisper_ctx, wparams, audio_buffer.data(), audio_buffer.size()) == 0) {
                    const int n_segments = whisper_full_n_segments(whisper_ctx);

                    std::string full_text;
                    for (int i = 0; i < n_segments; ++i) {
                        const char* text = whisper_full_get_segment_text(whisper_ctx, i);
                        full_text += text;
                    }

                    // Trim whitespace
                    full_text.erase(0, full_text.find_first_not_of(" \t\n\r"));
                    full_text.erase(full_text.find_last_not_of(" \t\n\r") + 1);

                    if (!full_text.empty()) {
                        processCommand(full_text);
                    }
                }

                // Keep last 0.5 seconds for continuity
                const int overlap_samples = WHISPER_SAMPLE_RATE_CUSTOM / 2;
                if (audio_buffer.size() > overlap_samples) {
                    audio_buffer.erase(audio_buffer.begin(), audio_buffer.end() - overlap_samples);
                } else {
                    audio_buffer.clear();
                }
            }
        }

        if (capture_handle != nullptr) {
            snd_pcm_close(capture_handle);
            capture_handle = nullptr;
        }
        std::cout << "Whisper recognition stopped." << std::endl;
    }
};

// Global audio detector instance
static std::unique_ptr<AudioDetector> audio_detector;

// Signal handler to catch Ctrl+C
void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Force exit!" << std::endl;
    _exit(0);  // Immediate exit, bypasses all cleanup
}

// --- Pose Detection Struct ---
struct PoseDetection {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kp_scores;
    long long tracking_id = -1;
};

// IoU calculation
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;
    if (intersection.empty()) return 0.0f;
    return static_cast<float>(intersection.area()) / (box1.area() + box2.area() - intersection.area());
}

// YOLO-Pose Detector Class
class YoloPoseDetector {
public:
    YoloPoseDetector(const std::string& model_path, const std::string& class_names_path) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            std::cerr << "ERROR: Could not load YOLO-Pose model from " << model_path << std::endl;
            exit(1);
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::ifstream ifs(class_names_path);
        if (!ifs.is_open()) {
            std::cerr << "ERROR: Could not open class names file: " << class_names_path << std::endl;
            exit(1);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            class_names.push_back(line);
        }
        std::cout << "YOLO-Pose model loaded from: " << model_path << std::endl;
        std::cout << "Loaded " << class_names.size() << " class names." << std::endl;
    }

    std::vector<PoseDetection> detect(const cv::Mat& frame) {
        std::vector<PoseDetection> detections;
        if (frame.empty()) return detections;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        cv::Mat output_data = outputs[0].reshape(1, outputs[0].size[1]);
        output_data = output_data.t();

        const int num_rows = output_data.rows;
        const int num_cols = output_data.cols;
        const int expected_min_cols = 5 + NUM_KEYPOINTS * 3;
        if(num_cols != expected_min_cols) return detections;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::vector<cv::Point2f>> all_keypoints;
        std::vector<std::vector<float>> all_kp_scores;

        for(int i = 0; i < num_rows; ++i) {
            float* data = (float*)output_data.row(i).data;
            float confidence = data[4];

            if(confidence >= CONFIDENCE_THRESHOLD && confidence > SCORE_THRESHOLD) {
                float x_center = data[0];
                float y_center = data[1];
                float box_width = data[2];
                float box_height = data[3];

                int x = static_cast<int>((x_center - 0.5 * box_width) * frame.cols / INPUT_WIDTH);
                int y = static_cast<int>((y_center - 0.5 * box_height) * frame.rows / INPUT_HEIGHT);
                int width = static_cast<int>(box_width * frame.cols / INPUT_WIDTH);
                int height = static_cast<int>(box_height * frame.rows / INPUT_HEIGHT);

                boxes.push_back(cv::Rect(x,y,width,height));
                confidences.push_back(confidence);
                class_ids.push_back(0);

                std::vector<cv::Point2f> current_keypoints;
                std::vector<float> current_kp_scores;
                for(int k=0; k<NUM_KEYPOINTS; ++k) {
                    int kp_offset = 5 + k*3;
                    if(kp_offset + 2 < num_cols) {
                        float kp_x = data[kp_offset];
                        float kp_y = data[kp_offset+1];
                        float kp_score = data[kp_offset+2];
                        current_keypoints.push_back(cv::Point2f(kp_x * frame.cols / INPUT_WIDTH, kp_y * frame.rows / INPUT_HEIGHT));
                        current_kp_scores.push_back(kp_score);
                    } else {
                        current_keypoints.push_back(cv::Point2f(0,0));
                        current_kp_scores.push_back(0.f);
                    }
                }
                all_keypoints.push_back(current_keypoints);
                all_kp_scores.push_back(current_kp_scores);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for(int idx : indices) {
            detections.push_back({class_ids[idx], confidences[idx], boxes[idx], all_keypoints[idx], all_kp_scores[idx]});
        }
        return detections;
    }

    const std::vector<std::string>& getClassNames() const {
        return class_names;
    }

private:
    cv::dnn::Net net;
    std::vector<std::string> class_names;

    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr float CONFIDENCE_THRESHOLD = 0.7f;
    static constexpr float SCORE_THRESHOLD = 0.6f;
    static constexpr float NMS_THRESHOLD = 0.45f;
    static constexpr int NUM_KEYPOINTS = 17;
};

// Global static vars for tracking
static long long tracked_person_id_global = -1;
static std::string current_robot_state_string_global = "IDLE_STANDING";
static std::string current_person_follow_status_global = "Not Detected";

static std::atomic<bool> is_action_in_progress(false);
static std::atomic<RobotState> current_robot_physical_state(IDLE_STANDING);

// Convert enum to string
std::string robotStateToString(RobotState state) {
    switch(state) {
        case IDLE_STANDING: return "IDLE_STANDING";
        case SITTING: return "SITTING";
        case STANDING_UP_IN_PROGRESS: return "STANDING_UP_IN_PROGRESS";
        case SITTING_DOWN_IN_PROGRESS: return "SITTING_DOWN_IN_PROGRESS";
        case FOLLOWING: return "FOLLOWING";
        case STOPPING_FOR_STILL_PERSON: return "STOPPING_FOR_STILL_PERSON";
        case LOST_PERSON_IDLE: return "LOST_PERSON_IDLE";
        case SPINNING: return "SPINNING";
        default: return "UNKNOWN";
    }
}

void performTurnAround180(unitree::robot::go2::SportClient& client) {
    if (is_action_in_progress.load()) return;
    is_action_in_progress.store(true);
    std::cout << "Executing 180-degree turn..." << std::endl;

    std::thread([&]() {
        const int TURN_DURATION_MS = 3200;
        auto start_time = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count() < TURN_DURATION_MS) {
            client.Move(0.0f, 0.0f, TURN_AROUND_SPEED);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        client.Move(0.0f, 0.0f, 0.0f);
        is_action_in_progress.store(false);
        std::cout << "180-degree turn completed!" << std::endl;
    }).detach();
}

void startContinuousSpin(unitree::robot::go2::SportClient& client) {
    if (is_spinning.load()) {
        std::cout << "Already spinning!" << std::endl;
        return;
    }

    is_spinning.store(true);
    current_robot_physical_state.store(SPINNING);
    std::cout << "Starting continuous spin... Say 'stop' to halt." << std::endl;

    std::thread([&]() {
        while (is_spinning.load() && running.load()) {
            client.Move(0.0f, 0.0f, CONTINUOUS_SPIN_SPEED);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        client.Move(0.0f, 0.0f, 0.0f);
        current_robot_physical_state.store(IDLE_STANDING);
        std::cout << "Continuous spin stopped." << std::endl;
    }).detach();
}

void stopContinuousSpin(unitree::robot::go2::SportClient& client) {
    if (is_spinning.load()) {
        std::cout << "Stopping continuous spin..." << std::endl;
        is_spinning.store(false);
    }
}

void performStandUp(unitree::robot::go2::SportClient& client) {
    if (is_action_in_progress.load()) return;
    is_action_in_progress.store(true);
    current_robot_physical_state.store(STANDING_UP_IN_PROGRESS);
    std::cout << "Robot executing StandUp()..." << std::endl;
    client.StandUp();
    std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
        is_action_in_progress.store(false);
        current_robot_physical_state.store(IDLE_STANDING);
        std::cout << "StandUp() completed. Robot now IDLE_STANDING." << std::endl;
    }).detach();
}

void performSit(unitree::robot::go2::SportClient& client) {
    if (is_action_in_progress.load()) return;
    is_action_in_progress.store(true);
    current_robot_physical_state.store(SITTING_DOWN_IN_PROGRESS);
    std::cout << "Robot executing Sit()..." << std::endl;
    client.Sit();
    std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
        is_action_in_progress.store(false);
        current_robot_physical_state.store(SITTING);
        std::cout << "Sit() completed. Robot now SITTING." << std::endl;
    }).detach();
}

// Robot command logic
void sendRobotCommand(unitree::robot::go2::SportClient& client, std::vector<PoseDetection>& detections, int image_width, int image_height) {
    static long long tracked_person_id = -1;
    static cv::Rect last_tracked_person_box;
    static cv::Point2f last_person_center_check = cv::Point2f(-1,-1);
    static float last_person_area_check = -1.f;
    static int frames_person_standing_still = 0;
    static int frames_person_too_close = 0;
    static int frames_person_lost = 0;

    PoseDetection* current_tracked_person_det = nullptr;
    current_robot_state_string_global = robotStateToString(current_robot_physical_state.load());

    if (audio_detector && audio_detector->isSpeechCommandReady()) {
        std::string voice_cmd = audio_detector->getLastCommand();
        RobotState current_state = current_robot_physical_state.load();

        std::cout << "Executing voice command: '" << voice_cmd << "'" << std::endl;

        if (voice_cmd == "sit") {
            if (current_state != SITTING && current_state != SITTING_DOWN_IN_PROGRESS) {
                std::cout << "Sitting down on voice command..." << std::endl;
                performSit(client);
            } else {
                std::cout << "Robot is already sitting." << std::endl;
            }
            return;
        }

        else if (voice_cmd == "stand") {
            if (current_state == SITTING || current_state == SITTING_DOWN_IN_PROGRESS) {
                std::cout << "Standing up on voice command..." << std::endl;
                performStandUp(client);
            } else {
                std::cout << "Robot already standing/active." << std::endl;
            }
            return;
        }

        else if (voice_cmd == "turn_around") {
            if (current_state != SITTING && !is_spinning.load()) {
                performTurnAround180(client);
            }
            return;
        }
        else if (voice_cmd == "spin") {
            if (current_state != SITTING && !is_spinning.load()) {
                startContinuousSpin(client);
            }
            return;
        }
        else if (voice_cmd == "damp" || voice_cmd == "balance" || voice_cmd == "recovery" ||
                 voice_cmd == "hello" || voice_cmd == "stretch" || voice_cmd == "dance" ||
                 voice_cmd == "flip" || voice_cmd == "jump") {
            if (voice_cmd == "damp") client.Damp();
            else if (voice_cmd == "balance") client.BalanceStand();
            else if (voice_cmd == "recovery") client.RecoveryStand();
            else if (voice_cmd == "hello") client.Hello();
            else if (voice_cmd == "stretch") client.Stretch();
            else if (voice_cmd == "dance") client.Dance1();
            else if (voice_cmd == "flip") client.FrontFlip();
            else if (voice_cmd == "jump") client.FrontJump();
            return;
        }

        else if (voice_cmd == "follow") {
            if (current_state == SITTING) {
                std::cout << "Standing up to follow..." << std::endl;
                performStandUp(client);
                return;
            }
            std::cout << "Follow mode enabled" << std::endl;
        }
        else if (voice_cmd == "stop") {
            std::cout << "Stop command" << std::endl;
            if (is_spinning.load()) {
                stopContinuousSpin(client);
                return;
            }
            client.Move(0, 0, 0);
            if (current_state == FOLLOWING) {
                current_robot_physical_state.store(IDLE_STANDING);
            }
        }
    }

    if (is_action_in_progress.load()) {
        std::cout << "Robot: Action in progress (" << current_robot_state_string_global << "), waiting..." << std::endl;
        client.Move(0,0,0);
        return;
    }

    bool person_detected_in_frame = false;
    if (!detections.empty()) {
        person_detected_in_frame = true;
        if (tracked_person_id == -1) {
            std::sort(detections.begin(), detections.end(), [](const PoseDetection& a, const PoseDetection& b) {
                return a.box.area() > b.box.area();
            });
            current_tracked_person_det = &detections[0];
            tracked_person_id = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            last_tracked_person_box = current_tracked_person_det->box;
            current_tracked_person_det->tracking_id = tracked_person_id;
            std::cout << "Started tracking new person with ID: " << tracked_person_id << std::endl;
            current_person_follow_status_global = "Tracking ID: " + std::to_string(tracked_person_id);
            frames_person_standing_still = 0;
            frames_person_too_close = 0;
        } else {
            float max_iou = 0.f;
            PoseDetection* best_match_det = nullptr;
            for (auto& det : detections) {
                float iou = calculateIoU(last_tracked_person_box, det.box);
                if (iou > max_iou && iou > 0.3f) {
                    max_iou = iou;
                    best_match_det = &det;
                }
            }
            if (best_match_det) {
                current_tracked_person_det = best_match_det;
                last_tracked_person_box = current_tracked_person_det->box;
                current_tracked_person_det->tracking_id = tracked_person_id;
                current_person_follow_status_global = "Tracking ID: " + std::to_string(tracked_person_id);
            } else {
                current_tracked_person_det = nullptr;
                std::cout << "Tracked person (ID: " << tracked_person_id << ") lost." << std::endl;
            }
        }
    } else {
        current_tracked_person_det = nullptr;
    }
    tracked_person_id_global = tracked_person_id;

    std::string desired_high_level_vision_state = "stop";

    if (current_tracked_person_det) {
        float normalized_person_height = static_cast<float>(current_tracked_person_det->box.height) / image_height;
        float normalized_person_center_x = (static_cast<float>(current_tracked_person_det->box.x) + current_tracked_person_det->box.width / 2.0f) / image_width;

        bool person_is_still = false;
        if (last_person_center_check.x != -1) {
            float dx = (current_tracked_person_det->box.x + current_tracked_person_det->box.width/2.0f) - last_person_center_check.x;
            float dy = (current_tracked_person_det->box.y + current_tracked_person_det->box.height/2.0f) - last_person_center_check.y;
            float d_area_ratio = std::abs(current_tracked_person_det->box.area() - last_person_area_check) / last_person_area_check;

            if (std::abs(dx) < 5.0f && std::abs(dy) < 5.0f && d_area_ratio < 0.05f) {
                person_is_still = true;
            }
        }

        last_person_center_check = cv::Point2f(current_tracked_person_det->box.x + current_tracked_person_det->box.width / 2.0f,
                                               current_tracked_person_det->box.y + current_tracked_person_det->box.height / 2.0f);
        last_person_area_check = static_cast<float>(current_tracked_person_det->box.area());

        if (current_robot_physical_state.load() == SITTING) {
            desired_high_level_vision_state = "follow";
            frames_person_too_close = 0;
            frames_person_standing_still = 0;
        } else if (normalized_person_height > PERSON_VERY_CLOSE_HEIGHT_RATIO) {
            desired_high_level_vision_state = "very_close_stop";
            frames_person_standing_still = 0;
            frames_person_too_close++;
        } else if (person_is_still && normalized_person_height > PERSON_FAR_HEIGHT_RATIO) {
            frames_person_standing_still++;
            if (frames_person_standing_still >= SIT_ON_STILL_DELAY_FRAMES) {
                desired_high_level_vision_state = "sit";
            } else {
                desired_high_level_vision_state = "stop";
            }
            frames_person_too_close = 0;
        } else if (normalized_person_height < PERSON_LOST_HEIGHT_RATIO) {
            desired_high_level_vision_state = "follow";
            frames_person_standing_still = 0;
            frames_person_too_close = 0;
        } else {
            desired_high_level_vision_state = "follow";
            frames_person_standing_still = 0;
            frames_person_too_close = 0;
        }
    } else {
        frames_person_standing_still = 0;
        frames_person_too_close = 0;

        if (tracked_person_id != -1 && current_robot_physical_state.load() != SITTING) {
            frames_person_lost++;

            // Wait for delay before sitting
            if (frames_person_lost >= LOST_PERSON_SIT_DELAY_FRAMES) {
                std::cout << "Person lost for " << (LOST_PERSON_SIT_DELAY_FRAMES / FRAME_RATE)
                          << " second(s), sitting down." << std::endl;
                tracked_person_id = -1;
                current_person_follow_status_global = "Person Lost. Sitting down.";
                performSit(client);
                frames_person_lost = 0;
                return;
            } else {
                // Still waiting, show countdown
                std::cout << "Person lost (" << frames_person_lost << "/"
                          << LOST_PERSON_SIT_DELAY_FRAMES << " frames). Waiting..." << std::endl;
                client.Move(0.f, 0.f, 0.f);  // Stop moving while waiting
                current_person_follow_status_global = "Person Lost - Waiting " +
                    std::to_string(LOST_PERSON_SIT_DELAY_FRAMES - frames_person_lost) + " frames...";
                desired_high_level_vision_state = "stop";
            }
        } else {
            // Reset counter when person is present or already sitting
            frames_person_lost = 0;
            desired_high_level_vision_state = "stop";
            current_person_follow_status_global = "Not Detected / Idle";
        }
    }

    float command_vx = 0.f;
    float command_vyaw = 0.f;
    RobotState physical_state = current_robot_physical_state.load();

    if (physical_state == IDLE_STANDING) {
        if (desired_high_level_vision_state == "follow") {
            current_robot_physical_state.store(FOLLOWING);
        } else if (desired_high_level_vision_state == "very_close_stop") {
            std::cout << "Stopping - person very close, waiting to sit" << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(STOPPING_FOR_STILL_PERSON);
        } else if (desired_high_level_vision_state == "sit") {
            performSit(client);
        } else if (desired_high_level_vision_state == "stop") {
            client.Move(0.f, 0.f, 0.f);
        } else if (desired_high_level_vision_state == "lost_person_sit") {
            performSit(client);
        }
    }
    else if (physical_state == FOLLOWING) {
        if (desired_high_level_vision_state == "follow" && current_tracked_person_det != nullptr) {
            float normalized_person_height = static_cast<float>(current_tracked_person_det->box.height) / image_height;
            float normalized_person_center_x = (static_cast<float>(current_tracked_person_det->box.x) + current_tracked_person_det->box.width/2.0f) / image_width;

            command_vx = MIN_FOLLOW_SPEED_X + (MAX_FOLLOW_SPEED_X - MIN_FOLLOW_SPEED_X) *
                         ((PERSON_CLOSE_HEIGHT_RATIO - normalized_person_height) / (PERSON_CLOSE_HEIGHT_RATIO - PERSON_FAR_HEIGHT_RATIO));
            command_vx = std::max(MIN_FOLLOW_SPEED_X, std::min(MAX_FOLLOW_SPEED_X, command_vx));

            int speed_level = 1;
            if (normalized_person_height < PERSON_FAR_HEIGHT_RATIO) speed_level = 2;
            else if (normalized_person_height < PERSON_MEDIUM_HEIGHT_RATIO) speed_level = 1;
            else speed_level = 0;
            client.SpeedLevel(speed_level);

            if (normalized_person_center_x < 0.5f - HORIZONTAL_CENTER_TOLERANCE)
                command_vyaw = TURN_SPEED_YAW;
            else if (normalized_person_center_x > 0.5f + HORIZONTAL_CENTER_TOLERANCE)
                command_vyaw = -TURN_SPEED_YAW;
            else
                command_vyaw = 0.f;

            std::string move_dir = (command_vyaw > 0) ? "left" : (command_vyaw < 0) ? "right" : "straight";
            std::cout << "Following: Vx=" << command_vx << ", Vyaw=" << command_vyaw << ", MoveDir=" << move_dir << ", SpeedLevel=" << speed_level << std::endl;
            client.Move(command_vx, 0.f, command_vyaw);
        } else if (desired_high_level_vision_state == "very_close_stop") {
            std::cout << "Person too close while following, stopping..." << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(STOPPING_FOR_STILL_PERSON);
        } else if (desired_high_level_vision_state == "stop") {
            std::cout << "Person stopped, robot stopping movement but staying in FOLLOWING" << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(STOPPING_FOR_STILL_PERSON);
        } else {
            std::cout << "Leaving FOLLOWING state to IDLE" << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(IDLE_STANDING);
        }
    }
    else if (physical_state == SITTING) {
        client.Move(0.f, 0.f, 0.f);

        // Optional: Show hint
        static auto last_hint_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_hint_time).count() > 10) {
            if (current_tracked_person_det != nullptr) {
                std::cout << "Robot sitting. Hold 'V' and say 'robot stand' to wake" << std::endl;
            }
            last_hint_time = now;
        }
    }
    else if (physical_state == STOPPING_FOR_STILL_PERSON) {
        static int frames_person_lost = 0;

        if (current_tracked_person_det == nullptr) {
            // Person lost while in stopping state
            frames_person_lost++;

            if (frames_person_lost >= LOST_PERSON_SIT_DELAY_FRAMES) {
                std::cout << "Person lost for " << (LOST_PERSON_SIT_DELAY_FRAMES / FRAME_RATE) << " seconds, sitting down." << std::endl;
                tracked_person_id = -1;
                current_person_follow_status_global = "Person Lost. Sitting down.";
                performSit(client);
                frames_person_lost = 0;
            } else {
                std::cout << "Person temporarily lost (" << frames_person_lost << "/" << LOST_PERSON_SIT_DELAY_FRAMES << " frames)..." << std::endl;
                client.Move(0.f, 0.f, 0.f);
                current_person_follow_status_global = "Person Temporarily Lost - Waiting...";
            }
        } else {
            // Person re-detected or still present
            frames_person_lost = 0;

            if (desired_high_level_vision_state == "sit" || desired_high_level_vision_state == "very_close_stop") {
                if (frames_person_too_close >= SIT_ON_CLOSE_DELAY_FRAMES || frames_person_standing_still >= SIT_ON_STILL_DELAY_FRAMES) {
                    performSit(client);
                } else {
                    std::cout << "Waiting to sit due to close or still person..." << std::endl;
                    client.Move(0.f, 0.f, 0.f);
                }
            } else if (desired_high_level_vision_state == "follow") {
                std::cout << "Person moved, resuming following..." << std::endl;
                frames_person_standing_still = 0;
                frames_person_too_close = 0;
                current_robot_physical_state.store(FOLLOWING);
            } else {
                std::cout << "Person moved or lost, going to IDLE..." << std::endl;
                client.Move(0.f, 0.f, 0.f);
                current_robot_physical_state.store(IDLE_STANDING);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_interface_name>" << std::endl;
        return 1;
    }

    signal(SIGINT, sigint_handler);

    unitree::robot::ChannelFactory::Instance()->Init(0, std::string(argv[1]));

    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f);
    sport_client.Init();
    sport_client_ptr = &sport_client;

    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    std::string model_dir = "/home/slam22/unitree_ws/src/unitree_sdk2/assets/models/yolov8/";
    YoloPoseDetector yolo_detector(model_dir + "yolo11n-pose.onnx", model_dir + "pose.names");

    // Initialize audio detector
    audio_detector = std::make_unique<AudioDetector>();
    audio_detector->start();

    std::cout << "\nInitializing robot..." << std::endl;
    std::cout << "Make sure the robot has enough space and is on a flat surface!" << std::endl;
    std::cout << "   Press Enter to make the robot stand up, or Ctrl+C to cancel..." << std::endl;
    std::cin.ignore();

    std::cout << "Commanding robot to stand up..." << std::endl;
    sport_client.StandUp();

    // Wait for stand up to complete
    std::cout << "   Waiting for robot to stand (3 seconds)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    current_robot_physical_state.store(IDLE_STANDING);
    std::cout << "Robot is standing and ready!" << std::endl;

    cv::namedWindow("Robot Camera Feed", cv::WINDOW_AUTOSIZE);

    std::cout << "\n----------------------------------------------------" << std::endl;
    std::cout << "Robot Control (YOLO-Pose Following + Push-to-Talk Voice)" << std::endl;
    std::cout << "Push-to-Talk: HOLD 'V' then say 'robot <command>'" << std::endl;
    std::cout << "    Movement: 'robot follow me', 'robot stop', 'robot sit', 'robot stand'" << std::endl;
    std::cout << "    Rotation: 'robot turn around', 'robot spin'" << std::endl;
    std::cout << "    Poses: 'robot damp', 'robot balance', 'robot recovery'" << std::endl;
    std::cout << "    Actions: 'robot hello', 'robot stretch', 'robot dance'" << std::endl;
    std::cout << "    Tricks: 'robot flip', 'robot jump'" << std::endl;
    std::cout << "Keyboard: 'm' = manual mode, WASD/QE = move" << std::endl;
    std::cout << "Vision: Auto-follows detected people when standing" << std::endl;
    std::cout << "Press Ctrl+C or ESC to stop." << std::endl;
    std::cout << "----------------------------------------------------\n" << std::endl;

    bool manual_control_mode = false;

    while (running.load()) {
        std::vector<uint8_t> image_data;
        int ret = video_client.GetImageSample(image_data);

        if (ret == 0 && !image_data.empty()) {
            cv::Mat raw_image = cv::imdecode(image_data, cv::IMREAD_COLOR);
            if (!raw_image.empty()) {
                std::vector<PoseDetection> detections = yolo_detector.detect(raw_image);

                std::vector<PoseDetection> filtered_detections;
                const auto& class_names = yolo_detector.getClassNames();
                for (const auto& det : detections) {
                    if (det.class_id < class_names.size() && class_names[det.class_id] == "person") {
                        filtered_detections.push_back(det);
                    }
                }

                const std::string instructions = "Hold 'V' to talk. Press 'm' for manual. WASD/QE = move/turn.";
                cv::putText(raw_image, instructions, cv::Point(10, raw_image.rows - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
                if (manual_control_mode) {
                    cv::putText(raw_image, "MANUAL CONTROL ACTIVE", cv::Point(10, 25),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                }
                if (push_to_talk_active.load()) {
                    cv::putText(raw_image, "LISTENING... (Release 'V' when done)", cv::Point(10, raw_image.rows - 50),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                }

                if (manual_control_mode) {
                    cv::putText(raw_image, "MANUAL CONTROL ACTIVE", cv::Point(10, 25),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                }
                int key = cv::waitKey(1);
                if (key == 'v' || key == 'V') {
                    if (!push_to_talk_active.load()) {
                        push_to_talk_active.store(true);
                        std::cout << "\nPUSH-TO-TALK ACTIVATED - Listening for 'robot' command..." << std::endl;
                    }
                } else if (push_to_talk_active.load() && key != -1) {
                    // Any other key releases push-to-talk
                    push_to_talk_active.store(false);
                    std::cout << "Push-to-talk released" << std::endl;
                }

                if (key == 27) {
                    std::cout << "\nESC pressed. Exiting..." << std::endl;
                    break;
                } else if (key == 'm' || key == 'M') {
                    manual_control_mode = !manual_control_mode;
                    std::cout << "Manual control mode toggled to " << (manual_control_mode ? "ON" : "OFF") << std::endl;
                    if (!manual_control_mode) {
                        sport_client.Move(0,0,0);
                        is_action_in_progress.store(false);
                        current_robot_physical_state.store(IDLE_STANDING);
                    }
                }

                if (manual_control_mode) {
                    float vx = 0.f;
                    float vyaw = 0.f;

                    if (key == 'w' || key == 'W') vx = MANUAL_MOVE_SPEED;
                    else if (key == 's' || key == 'S') vx = -MANUAL_MOVE_SPEED;

                    if (key == 'a' || key == 'A' || key == 'q' || key == 'Q') vyaw = MANUAL_TURN_SPEED;
                    else if (key == 'd' || key == 'D' || key == 'e' || key == 'E') vyaw = -MANUAL_TURN_SPEED;

                    if (vx != 0.f || vyaw != 0.f) {
                        is_action_in_progress.store(false);
                        current_robot_physical_state.store(IDLE_STANDING);
                        sport_client.Move(vx, 0.f, vyaw);
                    } else {
                        sport_client.Move(0.f, 0.f, 0.f);
                    }
                } else {
                    sendRobotCommand(sport_client, filtered_detections, raw_image.cols, raw_image.rows);
                }

                for (const auto& det : filtered_detections) {
                    cv::Scalar box_color = (det.tracking_id != -1 && det.tracking_id == tracked_person_id_global) ? cv::Scalar(255,0,0) : cv::Scalar(0,255,0);

                    cv::rectangle(raw_image, det.box, box_color, 2);
                    std::string label = class_names[det.class_id] + ": " + std::to_string(det.confidence).substr(0,4);
                    if (det.tracking_id != -1) label += " (ID:" + std::to_string(det.tracking_id) + ")";
                    cv::putText(raw_image, label, cv::Point(det.box.x, det.box.y-10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);

                    for (size_t i = 0; i < det.keypoints.size(); ++i) {
                        if (det.kp_scores[i] > 0.3f) {
                            cv::circle(raw_image, det.keypoints[i], 3, cv::Scalar(0,0,255), -1);
                        }
                    }
                }

                cv::putText(raw_image, "Robot State: " + current_robot_state_string_global, cv::Point(10,30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2, cv::LINE_AA);
                cv::putText(raw_image, "Person Status: " + current_person_follow_status_global, cv::Point(10,60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2, cv::LINE_AA);

                cv::imshow("Robot Camera Feed", raw_image);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (audio_detector) {
        audio_detector->stop();
    }

    cv::destroyAllWindows();
    std::cout << "Program exiting." << std::endl;
    return 0;
}