#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm> // For std::sort, std::max

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

// Global SportClient pointer to be accessible by signal handler
unitree::robot::go2::SportClient* sport_client_ptr = nullptr;

// --- Robot Control Constants ---
const float FOLLOW_SPEED_X = 0.3f;   // m/s forward speed when following
const float TURN_SPEED_YAW = 0.5f;   // rad/s turn speed
const float STOP_SPEED_X = 0.0f;
const float STOP_SPEED_YAW = 0.0f;

// Vision-based thresholds (normalized to image dimensions)
const float PERSON_TOO_CLOSE_HEIGHT_RATIO = 0.7f; // If person bbox height > 70% of image height
const float PERSON_TOO_FAR_HEIGHT_RATIO   = 0.2f; // If person bbox height < 20% of image height
const float RESUME_FOLLOW_HEIGHT_RATIO    = 0.4f; // If person bbox height > 40% when sitting, stand up and follow
const float HORIZONTAL_CENTER_TOLERANCE   = 0.15f; // If person bbox center_x is within +/- 15% of image center

// Delay for sitting/standing actions
const int FRAME_RATE = 100; // Approx. 100 FPS due to 10ms sleep
const int SIT_DELAY_FRAMES = 5 * FRAME_RATE; // 5 seconds of being too close to sit
const int ACTION_WAIT_MS = 1000; // 1 second wait after stand/sit commands

// Signal handler for Ctrl+C (SIGINT)
void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Stopping robot and exiting." << std::endl;
    if (sport_client_ptr) {
        sport_client_ptr->StopMove(); // Ensure robot stops on exit
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        sport_client_ptr->StandDown(); // Make it sit down on exit
    }
    running.store(false);
}

// --- YOLO-Pose Inference Structures ---
struct PoseDetection {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kp_scores;
    long long tracking_id = -1; // Added for simple tracking
};

// Calculate IoU (Intersection over Union) for two bounding boxes
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;
    float iou = (float)intersection.area() / (box1.area() + box2.area() - intersection.area());
    return iou;
}

class YoloPoseDetector {
public:
    YoloPoseDetector(const std::string& model_path, const std::string& class_names_path) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            std::cerr << "ERROR: Could not load YOLO-Pose ONNX model from " << model_path << std::endl;
            exit(1);
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Or DNN_TARGET_CUDA if you have GPU setup

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
            std::cerr << "WARNING: Expected 'pose.names' to contain only 'person' class for this pose model." << std::endl;
        }
    }

    std::vector<PoseDetection> detect(const cv::Mat& frame) {
        std::vector<PoseDetection> detections;
        if (frame.empty()) {
            std::cerr << "Input frame is empty for detection." << std::endl;
            return detections;
        }

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                               cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        cv::Mat output_data = outputs[0].reshape(1, outputs[0].size[1]);
        output_data = output_data.t();
        
        const int num_rows = output_data.rows;
        const int num_cols = output_data.cols;

        const int expected_min_cols = 5 + NUM_KEYPOINTS * 3;
        if (num_cols != expected_min_cols) {
            std::cerr << "WARNING: Unexpected output column size from pose model! Expected " << expected_min_cols << ", got " << num_cols << std::endl;
            return detections;
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::vector<cv::Point2f>> all_keypoints;
        std::vector<std::vector<float>> all_kp_scores;

        for (int i = 0; i < num_rows; ++i) {
            float* data = (float*)output_data.row(i).data;
            float confidence = data[4];

            if (confidence >= CONFIDENCE_THRESHOLD) {
                int class_id_val = 0; // Assuming only 'person' class (ID 0)
                double max_class_score = confidence; 

                if (max_class_score > SCORE_THRESHOLD) {
                    float x_center = data[0];
                    float y_center = data[1];
                    float box_width = data[2];
                    float box_height = data[3];

                    int x = static_cast<int>((x_center - 0.5 * box_width) * frame.cols / INPUT_WIDTH);
                    int y = static_cast<int>((y_center - 0.5 * box_height) * frame.rows / INPUT_HEIGHT);
                    int width = static_cast<int>(box_width * frame.cols / INPUT_WIDTH);
                    int height = static_cast<int>(box_height * frame.rows / INPUT_HEIGHT);

                    boxes.push_back(cv::Rect(x, y, width, height));
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id_val);

                    std::vector<cv::Point2f> current_keypoints;
                    std::vector<float> current_kp_scores;
                    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                        int kp_offset = 5 + k * 3;
                        if (kp_offset + 2 < num_cols) { 
                            float kp_x = data[kp_offset];
                            float kp_y = data[kp_offset + 1];
                            float kp_score = data[kp_offset + 2];

                            current_keypoints.push_back(cv::Point2f(kp_x * frame.cols / INPUT_WIDTH, kp_y * frame.rows / INPUT_HEIGHT));
                            current_kp_scores.push_back(kp_score);
                        } else {
                            current_keypoints.push_back(cv::Point2f(0,0));
                            current_kp_scores.push_back(0.0f);
                        }
                    }
                    all_keypoints.push_back(current_keypoints);
                    all_kp_scores.push_back(current_kp_scores);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
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

    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float CONFIDENCE_THRESHOLD = 0.7f;
    const float SCORE_THRESHOLD = 0.6f;
    const float NMS_THRESHOLD = 0.45f;
    const int NUM_KEYPOINTS = 17;
};

// --- Gesture Recognition Placeholder (uses PoseDetection) ---
std::string recognizeHandGestureFromPose(const PoseDetection& person_detection) {
    if (!person_detection.keypoints.empty() && person_detection.kp_scores.size() >= 11) {
        float left_wrist_score = person_detection.kp_scores[9];
        float right_wrist_score = person_detection.kp_scores[10];
        
        if (left_wrist_score > 0.5 || right_wrist_score > 0.5) {
            return "hand_visible";
        }
    }
    return "no_gesture";
}

// Global static variables for sendRobotCommand (accessed by main for drawing)
// Declared static at global scope for visibility in main's drawing loop
static long long tracked_person_id_global = -1;
static std::string current_robot_state_global = "stopped_standing";

// --- Robot Command Logic (Updated for following, turning, sitting) ---
void sendRobotCommand(unitree::robot::go2::SportClient& client, std::vector<PoseDetection>& detections, int image_width, int image_height) {
    // Current robot state: "sitting", "stopped_standing", "standing_up", "sitting_down", "following"
    static std::string current_robot_state = "stopped_standing"; 
    current_robot_state_global = current_robot_state; // Update global for main's drawing

    // Tracking variables for the identified person
    static long long tracked_person_id = -1; // Unique ID for the person being followed
    static cv::Rect last_tracked_person_box; // Bounding box of the last tracked person
    static int frames_person_too_close = 0; // Counter for continuous frames person is too close
    
    // Flags for ongoing actions (used internally by this function)
    static bool is_standing_up_in_progress = false;
    static bool is_sitting_down_in_progress = false;

    // A pointer to the detection of the person currently being tracked
    PoseDetection* current_tracked_person_det = nullptr;

    // --- Person Tracking Logic ---
    if (tracked_person_id == -1) {
        // No person currently tracked, find the best candidate (largest bounding box)
        if (!detections.empty()) {
            std::sort(detections.begin(), detections.end(), [](const PoseDetection& a, const PoseDetection& b) {
                return a.box.area() > b.box.area(); // Sort by largest area first
            });
            current_tracked_person_det = &detections[0];
            tracked_person_id = std::chrono::high_resolution_clock::now().time_since_epoch().count(); // Assign unique ID
            last_tracked_person_box = current_tracked_person_det->box;
            current_tracked_person_det->tracking_id = tracked_person_id; // Assign ID to the detected person
            std::cout << "Started tracking new person with ID: " << tracked_person_id << std::endl;
        }
    } else {
        // Person is already being tracked, try to re-associate with current detections
        float max_iou = 0.0f;
        PoseDetection* best_match_det = nullptr;

        for (auto& det : detections) {
            float iou = calculateIoU(last_tracked_person_box, det.box);
            // IoU threshold: If a detection overlaps significantly with the last tracked box, it's likely the same person.
            if (iou > max_iou && iou > 0.3f) { 
                max_iou = iou;
                best_match_det = &det;
            }
        }

        if (best_match_det) { // If a good match found (max_iou > 0.3f already verified)
            current_tracked_person_det = best_match_det;
            last_tracked_person_box = current_tracked_person_det->box; // Update box for next frame
            current_tracked_person_det->tracking_id = tracked_person_id; // Assign ID to current frame's detection
        } else {
            // Tracked person lost (no sufficient overlap found)
            std::cout << "Tracked person (ID: " << tracked_person_id << ") lost." << std::endl;
            tracked_person_id = -1; // Reset tracking
            last_tracked_person_box = cv::Rect(); // Reset box
        }
    }
    tracked_person_id_global = tracked_person_id; // Update global for main's drawing

    // --- High-Level State Determination ---
    std::string desired_high_level_state = "stop"; // Default to stop if no tracked person

    if (current_tracked_person_det) {
        float normalized_person_height = static_cast<float>(current_tracked_person_det->box.height) / image_height;
        float normalized_person_center_x = (static_cast<float>(current_tracked_person_det->box.x) + current_tracked_person_det->box.width / 2.0f) / image_width;

        // Visual feedback for debugging
        // std::cout << "Tracked Person (ID: " << tracked_person_id << "): HeightRatio=" << normalized_person_height 
        //           << ", CenterX=" << normalized_person_center_x << ", RobotState=" << current_robot_state << std::endl;

        if (normalized_person_height > PERSON_TOO_CLOSE_HEIGHT_RATIO) {
            // Person is too close
            if (current_robot_state == "sitting") {
                desired_high_level_state = "sit"; // Stay sitting if already sitting
            } else {
                // If not sitting, count frames for delay
                frames_person_too_close++;
                if (frames_person_too_close >= SIT_DELAY_FRAMES) {
                    desired_high_level_state = "sit"; // Time to sit
                } else {
                    desired_high_level_state = "stop"; // Stop while waiting to sit
                }
            }
        } else if (current_robot_state == "sitting") {
            // Robot is sitting, check if it should stand up and follow
            if (normalized_person_height < RESUME_FOLLOW_HEIGHT_RATIO) { // Person moved sufficiently far away
                desired_high_level_state = "follow";
                frames_person_too_close = 0; // Reset counter
            } else {
                desired_high_level_state = "sit"; // Stay sitting if person still within "sit" comfort zone
            }
        } else if (normalized_person_height < PERSON_TOO_FAR_HEIGHT_RATIO) {
            // Person too far or almost lost, stop
            desired_high_level_state = "stop";
            frames_person_too_close = 0; // Reset counter
        } else {
            // Person within follow range, and not sitting
            desired_high_level_state = "follow"; 
            frames_person_too_close = 0; // Reset counter
        }
    } else {
        // No person detected or tracked person lost
        frames_person_too_close = 0; // Reset counter
        desired_high_level_state = "stop"; 
    }

    // --- Robot Action Execution (State Machine) ---
    float command_vx = STOP_SPEED_X;
    float command_vyaw = STOP_SPEED_YAW;

    // Block new commands if an action is in progress
    if (is_standing_up_in_progress || is_sitting_down_in_progress) {
        return; 
    }

    if (desired_high_level_state == "sit") {
        if (current_robot_state != "sitting") {
            std::cout << "Robot Command: Sitting Down (Person too close for " << SIT_DELAY_FRAMES/FRAME_RATE << "s)" << std::endl;
            client.StandDown();
            is_sitting_down_in_progress = true;
            current_robot_state = "sitting_down"; // Transition state
            std::thread([&]() { // Lambda to wait in a separate thread
                std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
                is_sitting_down_in_progress = false;
                current_robot_state = "sitting"; // Final state after sitting
            }).detach(); 
            client.Move(STOP_SPEED_X, 0.0f, STOP_SPEED_YAW); // Ensure it stops moving before sitting
        }
    } else if (desired_high_level_state == "stop") {
        if (current_robot_state != "stopped_standing") { 
            if (current_robot_state == "sitting") {
                // If robot needs to stop but is sitting, it needs to stand up first
                std::cout << "Robot Command: Standing Up to stop (Person too far / lost)" << std::endl;
                client.StandUp();
                is_standing_up_in_progress = true;
                current_robot_state = "standing_up"; // Transition state
                std::thread([&]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
                    is_standing_up_in_progress = false;
                    current_robot_state = "stopped_standing"; // Final state after standing
                }).detach();
            } else { // If not sitting, just stop movement
                std::cout << "Robot Command: Stopping Motion (No tracked person)" << std::endl;
                client.StopMove();
                current_robot_state = "stopped_standing";
            }
        }
    } else if (desired_high_level_state == "follow") {
        // If robot needs to follow but is sitting, it must stand up first
        if (current_robot_state == "sitting") {
            std::cout << "Robot Command: Standing Up to follow (Person moved away)" << std::endl;
            client.StandUp();
            is_standing_up_in_progress = true;
            current_robot_state = "standing_up"; // Transition state
            std::thread([&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
                is_standing_up_in_progress = false;
                current_robot_state = "following"; // Will enter following state on next loop iteration
            }).detach();
        }
        
        // If standing or already following, calculate and send move commands
        if (current_robot_state == "stopped_standing" || current_robot_state == "following") {
            command_vx = FOLLOW_SPEED_X;
            float center_x = (static_cast<float>(current_tracked_person_det->box.x) + current_tracked_person_det->box.width / 2.0f) / image_width;

            if (center_x < 0.5f - HORIZONTAL_CENTER_TOLERANCE) {
                command_vyaw = TURN_SPEED_YAW; // Turn left
            } else if (center_x > 0.5f + HORIZONTAL_CENTER_TOLERANCE) {
                command_vyaw = -TURN_SPEED_YAW; // Turn right
            } else {
                command_vyaw = STOP_SPEED_YAW; // Go straight
            }
            
            // Only send new move command if different from last effective command or not yet 'following'
            // (Note: This simplified comparison doesn't use actual robot velocity feedback)
            if (current_robot_state != "following" || command_vx != 0.0f || command_vyaw != 0.0f) { // If moving or starting to move
                 std::string motion_desc = "straight";
                 if (command_vyaw > 0) motion_desc = "left";
                 else if (command_vyaw < 0) motion_desc = "right";
                 std::cout << "Robot Command: Following Person (" << motion_desc << ")" << std::endl;
                 client.Move(command_vx, 0.0f, command_vyaw);
                 current_robot_state = "following";
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

    cv::namedWindow("Robot Camera Feed", cv::WINDOW_AUTOSIZE);

    std::cout << "Robot Control (YOLO-Pose Following) Initialized. Press Ctrl+C or ESC to stop." << std::endl;
    std::cout << "Robot will follow, turn, or sit based on person's position/distance." << std::endl;
    std::cout << "Video feed with person detections and keypoints will appear." << std::endl;

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

                sendRobotCommand(sport_client, filtered_detections, raw_image.cols, raw_image.rows);

                // --- Draw filtered detections and keypoints ---
                for (const auto& det : filtered_detections) {
                    cv::Scalar box_color = cv::Scalar(0, 255, 0); // Default green
                    // If this detection is the tracked person, draw in blue
                    if (det.tracking_id != -1 && det.tracking_id == tracked_person_id_global) { 
                        box_color = cv::Scalar(255, 0, 0); // Blue for tracked person
                    }

                    // Draw bounding box
                    cv::rectangle(raw_image, det.box, box_color, 2);
                    std::string label = class_names[det.class_id] + ": " + std::to_string(det.confidence).substr(0, 4);
                    if (det.tracking_id != -1) {
                         label += " (ID:" + std::to_string(det.tracking_id) + ")";
                    }
                    cv::putText(raw_image, label, cv::Point(det.box.x, det.box.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);

                    // Draw keypoints
                    for (size_t i = 0; i < det.keypoints.size(); ++i) {
                        if (det.kp_scores[i] > 0.3) {
                             cv::circle(raw_image, det.keypoints[i], 3, cv::Scalar(0, 0, 255), -1); // Red circles
                        }
                    }
                }
                
                // Print current robot state for debugging on video feed
                cv::putText(raw_image, "Robot State: " + current_robot_state_global, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

                cv::imshow("Robot Camera Feed", raw_image);
            } else {
                std::cerr << "Error: Failed to decode image data." << std::endl;
            }
        } else {
            std::cerr << "Error getting image sample: " << ret << ". Image data empty: " << image_data.empty() << std::endl;
        }

        int key = cv::waitKey(1);
        if (key == 27) { // ESC key
            running.store(false);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Closing video window..." << std::endl;
    cv::destroyAllWindows();
    std::cout << "Exiting program." << std::endl;

    return 0;
}