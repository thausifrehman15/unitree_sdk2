#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <fstream> // For reading class names
#include <algorithm> // For std::remove_if

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
#include <opencv2/dnn/dnn.hpp> // For DNN module

// Global variable to control the main loop
std::atomic<bool> running(true);

// Global SportClient pointer to be accessible by signal handler
unitree::robot::go2::SportClient* sport_client_ptr = nullptr;

// --- Robot Control Helpers ---
const float MOVE_SPEED_X = 0.3f;
const float ROTATE_SPEED = 0.5f; // Not used for this specific task, but kept for reference

// Signal handler for Ctrl+C (SIGINT)
void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Stopping robot and exiting." << std::endl;
    if (sport_client_ptr) {
        sport_client_ptr->StopMove(); // Ensure robot stops on exit
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // sport_client_ptr->StandDown(); // Optional: Make it sit down on exit
    }
    running.store(false);
}

// --- YOLOv8 Inference Class ---
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, const std::string& class_names_path) {
        // Load the model
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            std::cerr << "ERROR: Could not load YOLOv8 ONNX model from " << model_path << std::endl;
            exit(1);
        }
        // Set backend and target (usually CPU is fine for initial testing, GPU if available and compiled with CUDA)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Or DNN_TARGET_CUDA if you have GPU setup

        // Load class names
        std::ifstream ifs(class_names_path);
        if (!ifs.is_open()) {
            std::cerr << "ERROR: Could not open class names file: " << class_names_path << std::endl;
            exit(1);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            class_names.push_back(line);
        }
        std::cout << "YOLOv8 model loaded from: " << model_path << std::endl;
        std::cout << "Loaded " << class_names.size() << " class names." << std::endl;
    }

    std::vector<Detection> detect(const cv::Mat& frame) {
        std::vector<Detection> detections;
        if (frame.empty()) {
            std::cerr << "Input frame is empty for detection." << std::endl;
            return detections;
        }

        // Pre-process the frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                               cv::Scalar(), true, false);
        net.setInput(blob);

        // Run inference
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Parse outputs
        cv::Mat output_data = outputs[0].reshape(1, outputs[0].size[2]); // (num_boxes, num_classes + 4)
        
        const int num_rows = output_data.rows;

        // Prepare lists for NMS
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        // Iterate through all detection predictions
        for (int i = 0; i < num_rows; ++i) {
            float* data = (float*)output_data.row(i).data;
            float confidence = data[4]; // Object confidence score

            // Using the current confidence threshold
            if (confidence >= CONFIDENCE_THRESHOLD) { 
                float* class_scores = data + 5; // Start of class probabilities
                cv::Mat scores_mat(1, class_names.size(), CV_32FC1, class_scores);
                cv::Point class_id_point;
                double max_class_score;
                
                // Get the class with the highest score
                cv::minMaxLoc(scores_mat, 0, &max_class_score, 0, &class_id_point);

                // Using the current score threshold
                if (max_class_score > SCORE_THRESHOLD) {
                    // Bounding box coordinates: x_center, y_center, width, height
                    float x_center = data[0];
                    float y_center = data[1];
                    float box_width = data[2];
                    float box_height = data[3];

                    // Convert to top-left corner and width/height for cv::Rect
                    int x = static_cast<int>((x_center - 0.5 * box_width) * frame.cols / INPUT_WIDTH);
                    int y = static_cast<int>((y_center - 0.5 * box_height) * frame.rows / INPUT_HEIGHT);
                    int width = static_cast<int>(box_width * frame.cols / INPUT_WIDTH);
                    int height = static_cast<int>(box_height * frame.rows / INPUT_HEIGHT);

                    boxes.push_back(cv::Rect(x, y, width, height));
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id_point.x);
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
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
    // Current thresholds (adjust as needed for your environment)
    const float CONFIDENCE_THRESHOLD = 0.7f; // Set in previous iteration
    const float SCORE_THRESHOLD = 0.6f;      // Set in previous iteration
    const float NMS_THRESHOLD = 0.45f;       // IoU threshold for NMS
};

// --- Robot Command Logic ---
// Maps detected objects to robot commands
void sendRobotCommand(unitree::robot::go2::SportClient& client, const std::vector<Detection>& detections, const std::vector<std::string>& class_names) {
    static std::string current_robot_state = "stopped"; // To avoid sending redundant commands
    std::string desired_action = "stop"; //lidar Default action if no relevant object detected

    bool person_present = false;
    
    // Iterate through filtered detections
    for (const auto& det : detections) {
        if (det.class_id < class_names.size()) {
            std::string class_name = class_names[det.class_id];
            
            if (class_name == "person") {
                person_present = true;
                desired_action = "go_forward"; // Robot moves forward if person is detected
                break; // Act on the first relevant detection found (person in this case)
            }
            // Add custom hand gesture classes here when you train your own model:
            // else if (class_name == "open_palm_gesture") { desired_action = "stop"; break; }
            // else if (class_name == "fist_gesture") { desired_action = "go_forward"; break; }
            // etc.
        }
    }

    // Execute robot commands based on `desired_action`
    if (desired_action == "go_forward") {
        if (current_robot_state != "forward") {
            std::cout << "Robot Command: Moving Forward (Person Detected)" << std::endl;
            client.Move(MOVE_SPEED_X, 0.0f, 0.0f);
            current_robot_state = "forward";
        }
    }
    // Other actions (like turn, sit, stand) are commented out as they'd require specific gesture triggers
    // else if (desired_action == "turn_left") { ... }
    // else if (desired_action == "sit_down") { ... }
    else if (desired_action == "stop") {
        if (current_robot_state != "stopped") {
            std::cout << "Robot Command: Stopping Motion (No Person Detected)" << std::endl;
            client.StopMove();
            current_robot_state = "stopped";
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_interface_name>" << std::endl;
        return 1;
    }

    // Setup signal handler for graceful exit on Ctrl+C.
    signal(SIGINT, sigint_handler);

    // Initialize the channel factory
    unitree::robot::ChannelFactory::Instance()->Init(0, std::string(argv[1]));
    
    // Initialize SportClient for robot control
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(10.0f);
    sport_client.Init();
    sport_client_ptr = &sport_client;

    // Initialize VideoClient for video capture
    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    // Initialize YOLO detector
    std::string model_dir = "/home/slam22/unitree_ws/src/unitree_sdk2/assets/models/yolov8/";
    // Using yolov8n.onnx and coco.names
    YoloDetector yolo_detector(model_dir + "yolov8n.onnx", model_dir + "coco.names"); 

    // OpenCV window for displaying video
    cv::namedWindow("Robot Camera Feed", cv::WINDOW_AUTOSIZE);

    std::cout << "Robot Control (Vision Only) Initialized. Press Ctrl+C or ESC to stop." << std::endl;
    std::cout << "Robot will move forward if a person is detected." << std::endl;
    std::cout << "Video feed with person detections will appear in a new window." << std::endl;

    while (running.load()) {
        std::vector<uint8_t> image_data;
        int ret = video_client.GetImageSample(image_data);

        if (ret == 0 && !image_data.empty()) {
            cv::Mat raw_image = cv::imdecode(image_data, cv::IMREAD_COLOR);

            if (!raw_image.empty()) {
                // Perform YOLO detection
                std::vector<Detection> detections = yolo_detector.detect(raw_image);

                // Filter detections for 'person' (ID 0 in COCO)
                std::vector<Detection> filtered_detections;
                const auto& class_names = yolo_detector.getClassNames();
                for (const auto& det : detections) {
                    if (det.class_id < class_names.size()) {
                        std::string class_name = class_names[det.class_id];
                        // Keep only "person" for now
                        if (class_name == "person"
                            // Uncomment and add your custom gesture classes here when you train your own model:
                            // || class_name == "open_palm_gesture"
                            // || class_name == "fist_gesture"
                            ) {
                            filtered_detections.push_back(det);
                        }
                    }
                }

                // Send command to the robot based on filtered detections
                sendRobotCommand(sport_client, filtered_detections, class_names);

                // --- Draw only filtered detections and display ---
                for (const auto& det : filtered_detections) {
                    cv::rectangle(raw_image, det.box, cv::Scalar(0, 255, 0), 2); // Green box
                    std::string label = class_names[det.class_id] + ": " + std::to_string(det.confidence).substr(0, 4);
                    cv::putText(raw_image, label, cv::Point(det.box.x, det.box.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }

                cv::imshow("Robot Camera Feed", raw_image);
            } else {
                std::cerr << "Error: Failed to decode image data." << std::endl;
            }
        } else {
            std::cerr << "Error getting image sample: " << ret << ". Image data empty: " << image_data.empty() << std::endl;
        }

        int key = cv::waitKey(1); // Wait 1ms for a key press
        if (key == 27) { // ESC key
            running.store(false);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // ~100 FPS frame rate limit
    }

    std::cout << "Closing video window..." << std::endl;
    cv::destroyAllWindows();
    std::cout << "Exiting program." << std::endl;

    return 0;
}