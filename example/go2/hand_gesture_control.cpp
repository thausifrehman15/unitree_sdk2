#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm> // For std::sort, std::max, std::min
#include <cmath>     // For std::abs
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
const int SIT_ON_STILL_DELAY_FRAMES = 3 * FRAME_RATE; // 3 seconds wait for stillness before sitting
const int ACTION_WAIT_MS = 1500;

// Manual control constants
const float MANUAL_MOVE_SPEED = 0.3f;
const float MANUAL_TURN_SPEED = 0.7f;

// --- Robot State Enum ---
enum RobotState { 
    IDLE_STANDING, 
    SITTING, 
    STANDING_UP_IN_PROGRESS, 
    SITTING_DOWN_IN_PROGRESS, 
    FOLLOWING, 
    STOPPING_FOR_STILL_PERSON,
    LOST_PERSON_IDLE
};

// Signal handler to catch Ctrl+C
void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Stopping robot and exiting." << std::endl;
    if (sport_client_ptr) {
        sport_client_ptr->StopMove();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        sport_client_ptr->Sit();
    }
    running.store(false);
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

// YOLO-Pose Detector Class (simplified with constants)
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
        if (class_names.size() != 1 || class_names[0] != "person") {
            std::cerr << "WARNING: Expected 'pose.names' to contain only 'person' class." << std::endl;
        }
    }

    std::vector<PoseDetection> detect(const cv::Mat& frame) {
        std::vector<PoseDetection> detections;
        if (frame.empty()) {
            std::cerr << "Input frame empty!" << std::endl;
            return detections;
        }

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
        if(num_cols != expected_min_cols) {
            std::cerr << "Unexpected output size: Expected " << expected_min_cols << ", got " << num_cols << std::endl;
            return detections;
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::vector<cv::Point2f>> all_keypoints;
        std::vector<std::vector<float>> all_kp_scores;

        for(int i = 0; i < num_rows; ++i) {
            float* data = (float*)output_data.row(i).data;
            float confidence = data[4];

            if(confidence >= CONFIDENCE_THRESHOLD) {
                // Only one class "person"
                if(confidence > SCORE_THRESHOLD) {
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
        default: return "UNKNOWN";
    }
}

// Asynchronous StandUp command
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

// Asynchronous Sit command
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

    PoseDetection* current_tracked_person_det = nullptr;

    current_robot_state_string_global = robotStateToString(current_robot_physical_state.load());

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

        // Fix stillness calculation using last center
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
            if (normalized_person_height < RESUME_FOLLOW_HEIGHT_RATIO) {
                desired_high_level_vision_state = "follow";
                frames_person_too_close = 0;
                frames_person_standing_still = 0;
            } else {
                desired_high_level_vision_state = "sit";
            }
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
            // Person far, move faster to catch up
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
            std::cout << "Tracked person lost, sitting down." << std::endl;
            tracked_person_id = -1;
            current_person_follow_status_global = "Person Lost. Sitting down.";
            desired_high_level_vision_state = "lost_person_sit";
        } else {
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

            // Speed interpolation
            command_vx = MIN_FOLLOW_SPEED_X +
                         (MAX_FOLLOW_SPEED_X - MIN_FOLLOW_SPEED_X) *
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
        } else {
            std::cout << "Leaving FOLLOWING state to IDLE" << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(IDLE_STANDING);
        }
    }
    else if (physical_state == SITTING) {
        if (desired_high_level_vision_state == "follow") {
            performStandUp(client);
        } else if (desired_high_level_vision_state == "stop" || desired_high_level_vision_state == "sit" || desired_high_level_vision_state == "very_close_stop" || desired_high_level_vision_state == "lost_person_sit") {
            client.Move(0.f, 0.f, 0.f);
        }
    }
    else if (physical_state == STOPPING_FOR_STILL_PERSON) {
        if (desired_high_level_vision_state == "sit" || desired_high_level_vision_state == "very_close_stop") {
            if (frames_person_too_close >= SIT_ON_CLOSE_DELAY_FRAMES || frames_person_standing_still >= SIT_ON_STILL_DELAY_FRAMES) {
                performSit(client);
            } else {
                std::cout << "Waiting to sit due to close or still person..." << std::endl;
                client.Move(0.f, 0.f, 0.f);
            }
        } else {
            std::cout << "Person moved, leaving stop state..." << std::endl;
            client.Move(0.f, 0.f, 0.f);
            current_robot_physical_state.store(IDLE_STANDING);
        }
    }
}

// ---- Main Function ----
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

    cv::namedWindow("Robot Camera Feed", cv::WINDOW_AUTOSIZE);

    std::cout << "Robot Control (YOLO-Pose Following) Initialized. Press Ctrl+C or ESC to stop." << std::endl;
    std::cout << "Robot will follow, turn, sit (delayed), and manage lost persons." << std::endl;
    std::cout << "Keyboard: 'm' toggle manual mode. WASD/QE to move in manual." << std::endl;

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

                // Overlay instructions
                const std::string instructions = "Press 'm' to toggle manual control. WASD/QE = move/turn.";
                cv::putText(raw_image, instructions, cv::Point(10, raw_image.rows - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
                if (manual_control_mode) {
                    cv::putText(raw_image, "MANUAL CONTROL ACTIVE", cv::Point(10, 25),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                }

                // Keyboard input handling
                int key = cv::waitKey(1);
                if (key == 27) { // ESC
                    running.store(false);
                } else if (key == 'm' || key == 'M') {
                    manual_control_mode = !manual_control_mode;
                    std::cout << "Manual control mode toggled to " << (manual_control_mode ? "ON" : "OFF") << std::endl;
                    if (!manual_control_mode) {
                        // When leaving manual, stop robot
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

                // Draw detections and keypoints
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

                // Show robot state and status
                cv::putText(raw_image, "Robot State: " + current_robot_state_string_global, cv::Point(10,30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2, cv::LINE_AA);
                cv::putText(raw_image, "Person Status: " + current_person_follow_status_global, cv::Point(10,60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2, cv::LINE_AA);

                cv::imshow("Robot Camera Feed", raw_image);
            } else {
                std::cerr << "Failed to decode image." << std::endl;
            }
        } else {
            std::cerr << "Error getting image sample: " << ret << ", empty data: " << image_data.empty() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cv::destroyAllWindows();
    std::cout << "Program exiting." << std::endl;
    return 0;
}
