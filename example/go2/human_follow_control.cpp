/**
 * @file human_follow_control.cpp
 * @brief Vision-based human following for the Unitree Go2 robot.
 * @details This program uses YOLO for person detection and OSNet for ReID-based tracking
 *          with gallery management for robust person re-identification across occlusions.
 */

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
#include <mutex>
#include <sstream>
#include <map>
#include <deque>
#include <limits>
#include <numeric>

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

// Forward declarations
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
float cosineSimilarity(const cv::Mat& a, const cv::Mat& b);

// --- Global State & Control Variables ---
std::atomic<bool> running(true);
unitree::robot::go2::SportClient* sport_client_ptr = nullptr;
std::atomic<bool> is_spinning(false);
static std::atomic<float> g_last_yaw_command{0.0f};
static std::atomic<bool> manual_control_active(false);

// Manual control state
static float manual_vx = 0.0f;
static float manual_vyaw = 0.0f;
static bool manual_moving = false;

// --- Robot Control Tuning Parameters ---
const float MIN_FOLLOW_SPEED_X = 0.6f;
const float MAX_FOLLOW_SPEED_X = 3.0f;
const float TURN_SPEED_YAW = 1.0f;

const float MANUAL_MOVE_SPEED = 0.5f;
const float MANUAL_TURN_SPEED = 0.8f;

const float PERSON_VERY_CLOSE_HEIGHT_RATIO = 0.95f;
const float PERSON_CLOSE_HEIGHT_RATIO      = 0.65f;
const float PERSON_MEDIUM_HEIGHT_RATIO     = 0.45f;
const float PERSON_FAR_HEIGHT_RATIO        = 0.25f;
const float PERSON_LOST_HEIGHT_RATIO       = 0.15f;
const float RESUME_FOLLOW_HEIGHT_RATIO     = 0.4f;
const float HORIZONTAL_CENTER_TOLERANCE    = 0.15f;

const int FRAME_RATE = 30;
const int SIT_ON_CLOSE_DELAY_FRAMES = 5 * FRAME_RATE;
const int SIT_ON_STILL_DELAY_FRAMES = 3 * FRAME_RATE;
const int LOST_PERSON_SIT_DELAY_FRAMES = 3 * FRAME_RATE;
const int ACTION_WAIT_MS = 1500;

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

// --- Utility Functions ---
inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;
    if (intersection.empty()) return 0.0f;
    return static_cast<float>(intersection.area()) / (box1.area() + box2.area() - intersection.area());
}

float cosineSimilarity(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.type() != CV_32F || b.type() != CV_32F) return 0.0f;
    return a.dot(b); // Already L2 normalized
}

// --- Kalman Filter for Motion Prediction ---
class KalmanTracker {
public:
    KalmanTracker(const cv::Rect& initial_bbox) {
        kf = cv::KalmanFilter(8, 4, 0);
        kf.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
        kf.transitionMatrix.at<float>(0, 4) = 1;
        kf.transitionMatrix.at<float>(1, 5) = 1;
        kf.transitionMatrix.at<float>(2, 6) = 1;
        kf.transitionMatrix.at<float>(3, 7) = 1;

        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

        kf.statePost.at<float>(0) = initial_bbox.x + initial_bbox.width / 2.0f;
        kf.statePost.at<float>(1) = initial_bbox.y + initial_bbox.height / 2.0f;
        kf.statePost.at<float>(2) = initial_bbox.width;
        kf.statePost.at<float>(3) = initial_bbox.height;
    }

    cv::Rect predict() {
        cv::Mat prediction = kf.predict();
        float cx = prediction.at<float>(0);
        float cy = prediction.at<float>(1);
        float w = std::max(1.0f, prediction.at<float>(2));
        float h = std::max(1.0f, prediction.at<float>(3));
        return cv::Rect(static_cast<int>(cx - w/2), static_cast<int>(cy - h/2), 
                       static_cast<int>(w), static_cast<int>(h));
    }

    void update(const cv::Rect& bbox) {
        cv::Mat measurement(4, 1, CV_32F);
        measurement.at<float>(0) = bbox.x + bbox.width / 2.0f;
        measurement.at<float>(1) = bbox.y + bbox.height / 2.0f;
        measurement.at<float>(2) = bbox.width;
        measurement.at<float>(3) = bbox.height;
        kf.correct(measurement);
    }

private:
    cv::KalmanFilter kf;
};

// --- OSNet ReID Feature Extractor ---
class ReIDExtractor {
public:
    ReIDExtractor(const std::string& model_path) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            std::cerr << "ERROR: Could not load OSNet model from " << model_path << std::endl;
            std::cerr << "Using fallback color histogram features." << std::endl;
            use_osnet = false;
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            use_osnet = true;
            std::cout << "OSNet ReID model loaded successfully." << std::endl;
        }
    }
    
    cv::Mat extract(const cv::Mat& frame, const cv::Rect& bbox) {
        cv::Rect safe_box = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_box.area() == 0) return cv::Mat();
        
        cv::Mat roi = frame(safe_box);
        
        if (use_osnet) {
            cv::Mat resized;
            cv::resize(roi, resized, cv::Size(128, 256));
            
            cv::Mat blob;
            cv::dnn::blobFromImage(resized, blob, 1.0/255.0, cv::Size(128, 256), 
                                   cv::Scalar(0.485, 0.456, 0.406), true, false);
            
            net.setInput(blob);
            cv::Mat embedding = net.forward();
            cv::Mat normalized;
            cv::normalize(embedding, normalized, 1.0, 0.0, cv::NORM_L2);
            return normalized.reshape(1, 1);
        } else {
            return extractColorHistogram(roi);
        }
    }
    
private:
    cv::dnn::Net net;
    bool use_osnet;
    
    cv::Mat extractColorHistogram(const cv::Mat& roi) {
        cv::Mat hsv;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
        int h_bins = 30, s_bins = 32;
        int histSize[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180};
        float s_ranges[] = {0, 256};
        const float* ranges[] = {h_ranges, s_ranges};
        int channels[] = {0, 1};
        cv::Mat hist;
        cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);
        return hist.reshape(1, 1);
    }
};

// --- Track with Embedding Gallery ---
struct Track {
    long long id;
    cv::Rect bbox;
    KalmanTracker kf;
    std::deque<cv::Mat> embedding_gallery;
    int hits;
    int age;
    int time_since_update;
    float score;
    
    static const int MAX_GALLERY_SIZE = 10;
    static const int MIN_HITS_FOR_CONFIRMATION = 3;

    Track(long long track_id, const cv::Rect& initial_bbox, const cv::Mat& embedding, float det_score)
        : id(track_id), bbox(initial_bbox), kf(initial_bbox),
          hits(1), age(1), time_since_update(0), score(det_score) {
        if (!embedding.empty()) {
            embedding_gallery.push_back(embedding.clone());
        }
    }

    void predict() {
        bbox = kf.predict();
        age++;
        time_since_update++;
    }

    void update(const cv::Rect& new_bbox, const cv::Mat& new_embedding, float det_score) {
        kf.update(new_bbox);
        bbox = new_bbox;
        score = det_score;
        hits++;
        time_since_update = 0;
        
        if (!new_embedding.empty()) {
            embedding_gallery.push_back(new_embedding.clone());
            if (embedding_gallery.size() > MAX_GALLERY_SIZE) {
                embedding_gallery.pop_front();
            }
        }
    }
    
    float matchScore(const cv::Mat& query_embedding) const {
        if (query_embedding.empty() || embedding_gallery.empty()) return 0.0f;
        
        float max_sim = 0.0f;
        for (const auto& stored_emb : embedding_gallery) {
            float sim = cosineSimilarity(stored_emb, query_embedding);
            max_sim = std::max(max_sim, sim);
        }
        return max_sim;
    }
    
    bool isConfirmed() const { return hits >= MIN_HITS_FOR_CONFIRMATION; }
};

// --- Advanced ReID-based Tracker with Matching Cascade ---
class ReIDTracker {
public:
    ReIDTracker() : next_id(1), max_time_lost(30), locked_track_id(-1) {}
    
    void setLockedTrackId(long long locked_id) { locked_track_id = locked_id; }
    
    std::vector<Track>& update(const std::vector<cv::Rect>& detections, 
                               const std::vector<cv::Mat>& embeddings,
                               const std::vector<float>& scores) {
        // Stage 1: Predict all tracks
        for (auto& track : tracks) {
            track.predict();
        }
        
        // Stage 2: Separate tracks into confirmed, tentative, and lost
        std::vector<int> confirmed_idx, tentative_idx, lost_idx;
        for (size_t i = 0; i < tracks.size(); i++) {
            if (tracks[i].time_since_update == 0 && tracks[i].isConfirmed()) {
                confirmed_idx.push_back(i);
            } else if (tracks[i].time_since_update == 0 && !tracks[i].isConfirmed()) {
                tentative_idx.push_back(i);
            } else if (tracks[i].time_since_update > 0) {
                lost_idx.push_back(i);
            }
        }
        
        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> track_matched(tracks.size(), false);
        
        // Stage 3: Matching Cascade - Level 1 (Confirmed tracks: IoU + Appearance)
        if (!confirmed_idx.empty() && !detections.empty()) {
            matchTracks(confirmed_idx, detections, embeddings, scores, 
                       det_matched, track_matched, 0.3f, 0.5f, 0.6f);
        }
        
        // Stage 4: Matching Cascade - Level 2 (Tentative tracks: IoU only)
        if (!tentative_idx.empty() && !detections.empty()) {
            matchTracks(tentative_idx, detections, embeddings, scores,
                       det_matched, track_matched, 0.4f, 0.0f, 0.5f);
        }
        
        // Stage 5: Matching Cascade - Level 3 (Lost tracks: Appearance only)
        if (!lost_idx.empty() && !detections.empty()) {
            matchTracksAppearanceOnly(lost_idx, detections, embeddings, scores,
                                     det_matched, track_matched, 0.6f);
        }
        
        // Stage 6: Create new tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); i++) {
            if (!det_matched[i]) {
                bool overlaps = false;
                for (const auto& track : tracks) {
                    if (calculateIoU(track.bbox, detections[i]) > 0.3f) {
                        overlaps = true;
                        break;
                    }
                }
                if (!overlaps) {
                    cv::Mat emb = (i < embeddings.size()) ? embeddings[i] : cv::Mat();
                    float scr = (i < scores.size()) ? scores[i] : 0.5f;
                    tracks.emplace_back(next_id++, detections[i], emb, scr);
                }
            }
        }
        
        // Stage 7: Remove dead tracks
        int effective_max_time_lost = max_time_lost;
        for (auto& track : tracks) {
            if (locked_track_id != -1 && track.id == locked_track_id) {
                effective_max_time_lost = 150; // 5 seconds @ 30fps
                if (track.time_since_update > effective_max_time_lost) {
                    std::cout << "Locked person (ID: " << locked_track_id << ") lost. Clearing." << std::endl;
                    locked_track_id = -1;
                }
                break;
            }
        }
        
        tracks.erase(
            std::remove_if(tracks.begin(), tracks.end(),
                [this, effective_max_time_lost](const Track& t) {
                    bool is_locked = (locked_track_id != -1 && t.id == locked_track_id);
                    int age_threshold = is_locked ? effective_max_time_lost : max_time_lost;
                    return t.time_since_update > age_threshold;
                }),
            tracks.end()
        );
        
        return tracks;
    }
    
    std::vector<Track>& getTracks() { return tracks; }
    
    Track* findTrackById(long long id) {
        for (auto& track : tracks) {
            if (track.id == id) return &track;
        }
        return nullptr;
    }

private:
    void matchTracks(const std::vector<int>& track_indices,
                    const std::vector<cv::Rect>& detections,
                    const std::vector<cv::Mat>& embeddings,
                    const std::vector<float>& scores,
                    std::vector<bool>& det_matched,
                    std::vector<bool>& track_matched,
                    float iou_weight,
                    float app_weight,
                    float threshold) {
        
        std::vector<std::vector<float>> cost_matrix(track_indices.size(), 
                                                    std::vector<float>(detections.size(), 1.0f));
        
        for (size_t i = 0; i < track_indices.size(); i++) {
            int track_idx = track_indices[i];
            bool is_locked = (locked_track_id != -1 && tracks[track_idx].id == locked_track_id);
            
            for (size_t j = 0; j < detections.size(); j++) {
                if (det_matched[j]) continue;
                
                float iou = calculateIoU(tracks[track_idx].bbox, detections[j]);
                float app_sim = 0.0f;
                
                if (j < embeddings.size() && !embeddings[j].empty()) {
                    app_sim = tracks[track_idx].matchScore(embeddings[j]);
                }
                
                // For locked track, prioritize IoU to prevent switching
                if (is_locked) {
                    cost_matrix[i][j] = 1.0f - (0.8f * iou + 0.2f * app_sim);
                } else {
                    cost_matrix[i][j] = 1.0f - (iou_weight * iou + app_weight * app_sim);
                }
            }
        }
        
        auto matches = greedyAssignment(cost_matrix, threshold);
        
        for (const auto& match : matches) {
            int track_idx = track_indices[match.first];
            int det_idx = match.second;
            
            cv::Mat emb = (det_idx < embeddings.size()) ? embeddings[det_idx] : cv::Mat();
            float scr = (det_idx < scores.size()) ? scores[det_idx] : 0.5f;
            
            tracks[track_idx].update(detections[det_idx], emb, scr);
            track_matched[track_idx] = true;
            det_matched[det_idx] = true;
        }
    }
    
    void matchTracksAppearanceOnly(const std::vector<int>& track_indices,
                                   const std::vector<cv::Rect>& detections,
                                   const std::vector<cv::Mat>& embeddings,
                                   const std::vector<float>& scores,
                                   std::vector<bool>& det_matched,
                                   std::vector<bool>& track_matched,
                                   float threshold) {
        
        std::vector<std::vector<float>> cost_matrix(track_indices.size(), 
                                                    std::vector<float>(detections.size(), 1.0f));
        
        for (size_t i = 0; i < track_indices.size(); i++) {
            int track_idx = track_indices[i];
            for (size_t j = 0; j < detections.size(); j++) {
                if (det_matched[j]) continue;
                
                if (j < embeddings.size() && !embeddings[j].empty()) {
                    float app_sim = tracks[track_idx].matchScore(embeddings[j]);
                    cost_matrix[i][j] = 1.0f - app_sim;
                }
            }
        }
        
        auto matches = greedyAssignment(cost_matrix, threshold);
        
        for (const auto& match : matches) {
            int track_idx = track_indices[match.first];
            int det_idx = match.second;
            
            cv::Mat emb = (det_idx < embeddings.size()) ? embeddings[det_idx] : cv::Mat();
            float scr = (det_idx < scores.size()) ? scores[det_idx] : 0.5f;
            
            tracks[track_idx].update(detections[det_idx], emb, scr);
            track_matched[track_idx] = true;
            det_matched[det_idx] = true;
        }
    }
    
    std::vector<std::pair<int, int>> greedyAssignment(const std::vector<std::vector<float>>& cost_matrix, 
                                                     float thresh) {
        if (cost_matrix.empty()) return {};
        
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();
        
        std::vector<std::pair<int, int>> matches;
        std::vector<bool> row_matched(rows, false);
        std::vector<bool> col_matched(cols, false);
        
        for (int iter = 0; iter < std::min(rows, cols); iter++) {
            float min_cost = 1e9f;
            int min_i = -1, min_j = -1;
            
            for (int i = 0; i < rows; i++) {
                if (row_matched[i]) continue;
                for (int j = 0; j < cols; j++) {
                    if (col_matched[j]) continue;
                    if (cost_matrix[i][j] < min_cost) {
                        min_cost = cost_matrix[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }
            
            if (min_cost < thresh && min_i >= 0 && min_j >= 0) {
                matches.push_back({min_i, min_j});
                row_matched[min_i] = true;
                col_matched[min_j] = true;
            } else {
                break;
            }
        }
        
        return matches;
    }
    
    std::vector<Track> tracks;
    long long next_id;
    int max_time_lost;
    long long locked_track_id;
};

// --- Performance Metrics ---
struct PerformanceMetrics {
    long long start_time_ns = 0;
    long long total_frames = 0;
    long long frames_with_track = 0;
    long long cumulative_yolo_ns = 0;
    long long cumulative_reid_ns = 0;
    long long cumulative_tracker_ns = 0;
    long long yolo_samples = 0;
    long long reid_samples = 0;
    long long tracker_samples = 0;
};

static PerformanceMetrics g_metrics;
static std::mutex g_metrics_mutex;

static void metrics_record_yolo(long long ns) {
    std::lock_guard<std::mutex> lk(g_metrics_mutex);
    g_metrics.cumulative_yolo_ns += ns;
    g_metrics.yolo_samples++;
}

static void metrics_record_reid(long long ns) {
    std::lock_guard<std::mutex> lk(g_metrics_mutex);
    g_metrics.cumulative_reid_ns += ns;
    g_metrics.reid_samples++;
}

static void metrics_record_tracker(long long ns) {
    std::lock_guard<std::mutex> lk(g_metrics_mutex);
    g_metrics.cumulative_tracker_ns += ns;
    g_metrics.tracker_samples++;
}

static void metrics_finalize() {
    std::lock_guard<std::mutex> lk(g_metrics_mutex);
    long long runtime_ns = now_ns() - g_metrics.start_time_ns;
    double runtime_s = runtime_ns / 1e9;
    
    std::cout << "\n========== PERFORMANCE REPORT ==========\n";
    std::cout << "Runtime: " << runtime_s << " s\n";
    std::cout << "Total Frames: " << g_metrics.total_frames << "\n";
    std::cout << "FPS: " << (g_metrics.total_frames / runtime_s) << "\n";
    
    if (g_metrics.yolo_samples > 0) {
        std::cout << "Avg YOLO: " << (g_metrics.cumulative_yolo_ns / g_metrics.yolo_samples / 1e6) << " ms\n";
    }
    if (g_metrics.reid_samples > 0) {
        std::cout << "Avg ReID: " << (g_metrics.cumulative_reid_ns / g_metrics.reid_samples / 1e6) << " ms\n";
    }
    if (g_metrics.tracker_samples > 0) {
        std::cout << "Avg Tracker: " << (g_metrics.cumulative_tracker_ns / g_metrics.tracker_samples / 1e6) << " ms\n";
    }
    std::cout << "========================================\n";
}

// --- Global Tracker Instance ---
static ReIDTracker reid_tracker;
static long long locked_person_id = -1;
static long long tracked_person_id_global = -1;
static std::string current_robot_state_string_global = "IDLE_STANDING";
static std::string current_person_follow_status_global = "Not Detected";
static std::atomic<bool> is_action_in_progress(false);
static std::atomic<RobotState> current_robot_physical_state(IDLE_STANDING);
static bool preserve_tracking_after_sit = false;
static bool waiting_for_trigger = false;  // New: Waiting for person to trigger stand-up

// --- Robot Actions ---
std::string robotStateToString(RobotState state) {
    switch(state) {
        case IDLE_STANDING: return "IDLE_STANDING";
        case SITTING: return "SITTING";
        case STANDING_UP_IN_PROGRESS: return "STANDING_UP";
        case SITTING_DOWN_IN_PROGRESS: return "SITTING_DOWN";
        case FOLLOWING: return "FOLLOWING";
        case STOPPING_FOR_STILL_PERSON: return "STOPPED";
        case LOST_PERSON_IDLE: return "LOST";
        case SPINNING: return "SPINNING";
        default: return "UNKNOWN";
    }
}

void performStandUp(unitree::robot::go2::SportClient& client) {
    if (is_action_in_progress.load()) return;
    is_action_in_progress.store(true);
    current_robot_physical_state.store(STANDING_UP_IN_PROGRESS);
    std::cout << "Standing up..." << std::endl;
    client.StandUp();
    std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
        is_action_in_progress.store(false);
        current_robot_physical_state.store(IDLE_STANDING);
        std::cout << "StandUp complete." << std::endl;
    }).detach();
}

void performSit(unitree::robot::go2::SportClient& client, bool preserve = false) {
    if (is_action_in_progress.load()) return;
    is_action_in_progress.store(true);
    current_robot_physical_state.store(SITTING_DOWN_IN_PROGRESS);
    preserve_tracking_after_sit = preserve;
    if (!preserve) {
        tracked_person_id_global = -1;
        locked_person_id = -1;
    }
    std::cout << "Sitting down..." << std::endl;
    client.Sit();
    std::thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(ACTION_WAIT_MS));
        is_action_in_progress.store(false);
        current_robot_physical_state.store(SITTING);
        std::cout << "Sit complete." << std::endl;
    }).detach();
}

// --- Robot Command Logic ---
void sendRobotCommand(unitree::robot::go2::SportClient& client, int image_width, int image_height) {
    static int frames_person_standing_still = 0;
    static int frames_person_too_close = 0;
    static int frames_locked_person_missing = 0;
    static int search_rotation_frames = 0;
    static long long last_command_time = 0;
    static float last_known_person_center_x = 0.5f;  // Track last position
    
    // Manual control overrides automatic following
    if (manual_control_active.load()) {
        // Send manual control commands continuously
        long long now = now_ns();
        if (manual_moving && (now - last_command_time > 30000000)) {
            client.Move(manual_vx, 0, manual_vyaw);
            last_command_time = now;
        }
        return;
    }
    
    Track* current_tracked_person = nullptr;
    current_robot_state_string_global = robotStateToString(current_robot_physical_state.load());

    if (is_action_in_progress.load()) {
        client.Move(0, 0, 0);
        return;
    }

    if (locked_person_id == -1) {
        current_person_follow_status_global = "No lock. Press 'L'";
        current_robot_physical_state.store(IDLE_STANDING);
        client.StopMove();
        frames_locked_person_missing = 0;
        search_rotation_frames = 0;
        last_known_person_center_x = 0.5f;
        return;
    }

    // Try to find the LOCKED person specifically
    current_tracked_person = reid_tracker.findTrackById(locked_person_id);
    
    // Check if locked person exists and is visible
    bool locked_person_visible = false;
    
    if (current_tracked_person) {
        // Person track exists - check if actually visible (not just predicted)
        if (current_tracked_person->time_since_update == 0) {
            // Person was detected this frame - check if bbox is mostly in frame
            cv::Rect frame_rect(0, 0, image_width, image_height);
            cv::Rect intersection = current_tracked_person->bbox & frame_rect;
            float visibility_ratio = static_cast<float>(intersection.area()) / current_tracked_person->bbox.area();
            
            if (visibility_ratio > 0.5f) {
                // Locked person is clearly visible
                locked_person_visible = true;
                frames_locked_person_missing = 0;
                search_rotation_frames = 0;
                tracked_person_id_global = current_tracked_person->id;
                
                // Store last known position
                last_known_person_center_x = (current_tracked_person->bbox.x + current_tracked_person->bbox.width/2.0f) / image_width;
                
                current_person_follow_status_global = "LOCKED & VISIBLE ID:" + std::to_string(locked_person_id);
                current_robot_physical_state.store(FOLLOWING);
            } else {
                // Person detected but mostly out of frame - consider as missing
                locked_person_visible = false;
                frames_locked_person_missing++;
                current_person_follow_status_global = "Locked person at edge of frame...";
                current_robot_physical_state.store(LOST_PERSON_IDLE);
            }
        } else {
            // Person track exists but is being predicted (not actually detected)
            locked_person_visible = false;
            frames_locked_person_missing++;
            current_person_follow_status_global = "Locked person not detected (predicting)...";
            current_robot_physical_state.store(LOST_PERSON_IDLE);
        }
    } else {
        // Locked person track doesn't exist at all
        locked_person_visible = false;
        frames_locked_person_missing++;
        tracked_person_id_global = -1;
        current_robot_physical_state.store(LOST_PERSON_IDLE);
    }
    
    // If locked person is NOT visible, initiate search sequence
    if (!locked_person_visible) {
        const int SEARCH_DELAY = 10;  // Changed from 2 * FRAME_RATE to 10 frames
        const int SEARCH_ROTATION_FRAMES = 10 * FRAME_RATE;  // 10 seconds of rotation
        
        if (frames_locked_person_missing < SEARCH_DELAY) {
            // Initial wait period - person just disappeared
            current_person_follow_status_global = "Locked person lost. Waiting... (" + 
                std::to_string(frames_locked_person_missing) + "/" + std::to_string(SEARCH_DELAY) + ")";
            client.StopMove();
            search_rotation_frames = 0;
        } else if (search_rotation_frames < SEARCH_ROTATION_FRAMES) {
            // Actively searching - rotate to find locked person
            int search_progress = (search_rotation_frames * 100) / SEARCH_ROTATION_FRAMES;
            current_person_follow_status_global = "Searching for locked person ID:" + 
                std::to_string(locked_person_id) + " (" + std::to_string(search_progress) + "%)";
            
            // Rotate in the direction where person was last seen
            // If person was on left side (< 0.5), rotate left (positive yaw)
            // If person was on right side (> 0.5), rotate right (negative yaw)
            float search_yaw = (last_known_person_center_x < 0.5f) ? 0.6f : -0.6f;
            g_last_yaw_command.store(search_yaw);
            client.Move(0, 0, search_yaw);
            search_rotation_frames++;
            current_robot_physical_state.store(SPINNING);
        } else {
            // Search complete - locked person not found after full rotation
            current_person_follow_status_global = "Locked person not found. Sitting down...";
            std::cout << "Locked person (ID: " << locked_person_id << ") not found after full rotation. Sitting down." << std::endl;
            
            performSit(client, false);
            locked_person_id = -1;
            reid_tracker.setLockedTrackId(-1);
            frames_locked_person_missing = 0;
            search_rotation_frames = 0;
            last_known_person_center_x = 0.5f;
            waiting_for_trigger = true;
        }
        return;
    }
    
    // Locked person IS visible - follow them
    if (current_tracked_person && locked_person_visible) {
        float norm_height = static_cast<float>(current_tracked_person->bbox.height) / image_height;
        float norm_center_x = (current_tracked_person->bbox.x + current_tracked_person->bbox.width/2.0f) / image_width;
        
        // Compute forward speed based on person's distance (height in frame)
        float speed_factor = (PERSON_CLOSE_HEIGHT_RATIO - norm_height) / 
                            (PERSON_CLOSE_HEIGHT_RATIO - PERSON_FAR_HEIGHT_RATIO);
        speed_factor = std::max(0.0f, std::min(1.0f, speed_factor));
        
        float command_vx = MIN_FOLLOW_SPEED_X + (MAX_FOLLOW_SPEED_X - MIN_FOLLOW_SPEED_X) * speed_factor;
        float command_vyaw = 0.0f;
        
        // Compute turning based on horizontal position
        if (norm_center_x < 0.5f - HORIZONTAL_CENTER_TOLERANCE) {
            command_vyaw = TURN_SPEED_YAW;  // Turn left
        } else if (norm_center_x > 0.5f + HORIZONTAL_CENTER_TOLERANCE) {
            command_vyaw = -TURN_SPEED_YAW;  // Turn right
        }
        
        // If turning a lot, reduce forward speed
        if (std::abs(command_vyaw) > 0.5f && std::abs(norm_center_x - 0.5f) > 0.25f) {
            command_vx *= 0.3f;  // Slow down when turning
        }
        
        // Stop if person is too close
        if (norm_height > PERSON_CLOSE_HEIGHT_RATIO) {
            command_vx = 0.0f;
            current_person_follow_status_global = "Locked person too close. Stopped.";
            current_robot_physical_state.store(STOPPING_FOR_STILL_PERSON);
        }
        
        // Stop if person is too far
        if (norm_height < PERSON_FAR_HEIGHT_RATIO) {
            command_vx = 0.0f;
            current_person_follow_status_global = "Locked person too far. Stopped.";
            current_robot_physical_state.store(STOPPING_FOR_STILL_PERSON);
        }
        
        g_last_yaw_command.store(command_vyaw);
        
        // Send command continuously
        long long now = now_ns();
        if (now - last_command_time > 30000000) {  // Send at ~30Hz
            client.Move(command_vx, 0, command_vyaw);
            last_command_time = now;
            
            // Debug output
            if (command_vx != 0.0f || command_vyaw != 0.0f) {
                std::cout << "Following ID:" << locked_person_id 
                         << " vx=" << command_vx << " vyaw=" << command_vyaw 
                         << " (person at " << (norm_center_x * 100) << "% width, " 
                         << (norm_height * 100) << "% height)" << std::endl;
            }
        }
    } else {
        // Safety fallback - stop if something went wrong
        client.StopMove();
    }
}

void sigint_handler(int sig) {
    std::cout << "\nCtrl+C detected. Generating report..." << std::endl;
    metrics_finalize();
    _exit(0);
}

// --- YOLO Detection ---
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLODetector {
public:
    YOLODetector(const std::string& model_path) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            std::cerr << "ERROR: Could not load YOLO model." << std::endl;
            exit(1);
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<Detection> detect(const cv::Mat& frame) {
        std::vector<Detection> detections;
        if (frame.empty()) return detections;
        
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
        
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        
        cv::Mat output = outputs[0].reshape(1, outputs[0].size[1]).t();
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        
        for (int i = 0; i < output.rows; i++) {
            float* data = (float*)output.row(i).data;
            float conf = data[4];
            
            if (conf >= 0.6f) {
                float x = data[0] * frame.cols / 640.0f;
                float y = data[1] * frame.rows / 640.0f;
                float w = data[2] * frame.cols / 640.0f;
                float h = data[3] * frame.rows / 640.0f;
                
                boxes.push_back(cv::Rect(x - w/2, y - h/2, w, h));
                confidences.push_back(conf);
            }
        }
        
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.6f, 0.45f, indices);
        
        for (int idx : indices) {
            detections.push_back({0, confidences[idx], boxes[idx]});
        }
        
        return detections;
    }

private:
    cv::dnn::Net net;
};

// --- Main ---
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_interface>" << std::endl;
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
    
    std::ifstream yolo_check(model_dir + "yolo11n-pose.onnx");
    if (!yolo_check.good()) {
        std::cerr << "ERROR: YOLO model not found!" << std::endl;
        std::cerr << "Expected: " << model_dir << "yolo11n-pose.onnx" << std::endl;
        std::cerr << "\nDownload with:" << std::endl;
        std::cerr << "  cd " << model_dir << std::endl;
        std::cerr << "  wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" << std::endl;
        return 1;
    }
    
    YOLODetector yolo_detector(model_dir + "yolo11n-pose.onnx");
    ReIDExtractor reid_extractor(model_dir + "osnet_x1_0.onnx");
    
    g_metrics.start_time_ns = now_ns();
    
    std::cout << "\nRobot will start in SITTING position." << std::endl;
    std::cout << "Wave your hand in front of the upward-facing camera to trigger stand-up." << std::endl;
    
    sport_client.Sit();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    current_robot_physical_state.store(SITTING);
    waiting_for_trigger = true;
    
    cv::namedWindow("Robot Feed", cv::WINDOW_AUTOSIZE);
    std::cout << "\n========== CONTROLS (Focus on video window!) ==========\n"
              << "TRACKING:\n"
              << "  L: Lock onto largest person\n"
              << "  U: Unlock\n"
              << "  R: Reset tracker\n"
              << "  M: Toggle Manual Control Mode\n"
              << "\nMANUAL CONTROL (when enabled):\n"
              << "  W: Forward\n"
              << "  S: Backward\n"
              << "  A: Rotate Left\n"
              << "  D: Rotate Right\n"
              << "  Space: Stop\n"
              << "\nOTHER:\n"
              << "  J: Manual sit/stand toggle\n"
              << "  ESC: Exit\n"
              << "=======================================================\n" << std::endl;
    
    int trigger_detection_frames = 0;
    const int TRIGGER_CONFIRMATION_FRAMES = 15;
    
    while (running.load()) {
        std::vector<uint8_t> image_data;
        if (video_client.GetImageSample(image_data) == 0 && !image_data.empty()) {
            cv::Mat frame = cv::imdecode(image_data, cv::IMREAD_COLOR);
            if (!frame.empty()) {
                g_metrics.total_frames++;
                
                long long t0 = now_ns();
                auto detections = yolo_detector.detect(frame);
                metrics_record_yolo(now_ns() - t0);
                
                std::vector<cv::Rect> boxes;
                std::vector<cv::Mat> embeddings;
                std::vector<float> scores;
                
                t0 = now_ns();
                for (const auto& det : detections) {
                    boxes.push_back(det.box);
                    embeddings.push_back(reid_extractor.extract(frame, det.box));
                    scores.push_back(det.confidence);
                }
                metrics_record_reid(now_ns() - t0);
                
                t0 = now_ns();
                reid_tracker.update(boxes, embeddings, scores);
                metrics_record_tracker(now_ns() - t0);
                
                // Trigger detection logic
                if (waiting_for_trigger && current_robot_physical_state.load() == SITTING) {
                    if (!boxes.empty()) {
                        trigger_detection_frames++;
                        current_person_follow_status_global = "Person detected! Hold position... (" + 
                            std::to_string(trigger_detection_frames) + "/" + 
                            std::to_string(TRIGGER_CONFIRMATION_FRAMES) + ")";
                        
                        if (trigger_detection_frames >= TRIGGER_CONFIRMATION_FRAMES) {
                            std::cout << "Trigger confirmed! Standing up..." << std::endl;
                            performStandUp(sport_client);
                            waiting_for_trigger = false;
                            trigger_detection_frames = 0;
                            current_person_follow_status_global = "Standing up. Press 'L' to lock.";
                        }
                    } else {
                        trigger_detection_frames = 0;
                        current_person_follow_status_global = "Waiting for trigger... Wave in front of camera.";
                    }
                }
                
                // Draw UI first
                for (const auto& t : reid_tracker.getTracks()) {
                    cv::Scalar color = (locked_person_id == t.id) ? cv::Scalar(255, 0, 0) : 
                                      t.isConfirmed() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                    int thick = (locked_person_id == t.id) ? 3 : 2;
                    cv::rectangle(frame, t.bbox, color, thick);
                    
                    std::string label = "ID:" + std::to_string(t.id);
                    if (locked_person_id == t.id) label = "LOCKED " + label;
                    
                    cv::putText(frame, label, 
                               cv::Point(t.bbox.x, t.bbox.y - 10), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                }
                
                // Status overlay
                cv::Scalar status_color = waiting_for_trigger ? cv::Scalar(0, 165, 255) : 
                                         manual_control_active.load() ? cv::Scalar(255, 165, 0) :
                                         cv::Scalar(0, 255, 255);
                
                cv::putText(frame, "State: " + current_robot_state_string_global, 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);
                cv::putText(frame, current_person_follow_status_global, 
                           cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);
                
                if (manual_control_active.load()) {
                    cv::putText(frame, "MANUAL CONTROL: W/A/S/D + Space", 
                               cv::Point(frame.cols/2 - 200, frame.rows - 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 165, 0), 2);
                    
                    // Show current command
                    std::string cmd_text = "Cmd: ";
                    if (manual_moving) {
                        if (manual_vx > 0) cmd_text += "FWD";
                        else if (manual_vx < 0) cmd_text += "BACK";
                        if (manual_vyaw > 0) cmd_text += " LEFT";
                        else if (manual_vyaw < 0) cmd_text += " RIGHT";
                    } else {
                        cmd_text += "STOPPED";
                    }
                    cv::putText(frame, cmd_text, 
                               cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                               cv::Scalar(255, 165, 0), 2);
                } else if (waiting_for_trigger) {
                    cv::putText(frame, "WAITING FOR TRIGGER", 
                               cv::Point(frame.cols/2 - 150, frame.rows - 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
                }
                
                cv::imshow("Robot Feed", frame);
                
                // Process keyboard from OpenCV window
                int key = cv::waitKey(1) & 0xFF;
                
                if (key == 27) { // ESC
                    break;
                } else if (key == 'l' || key == 'L') {
                    if (current_robot_physical_state.load() == IDLE_STANDING || 
                        current_robot_physical_state.load() == FOLLOWING) {
                        Track* best = nullptr;
                        float max_area = 0;
                        for (auto& t : reid_tracker.getTracks()) {
                            if (t.isConfirmed() && t.time_since_update == 0 && t.bbox.area() > max_area) {
                                max_area = t.bbox.area();
                                best = &t;
                            }
                        }
                        if (best) {
                            locked_person_id = best->id;
                            reid_tracker.setLockedTrackId(locked_person_id);
                            manual_control_active.store(false);
                            manual_moving = false;
                            sport_client.StopMove();
                            std::cout << "Locked ID: " << locked_person_id << " (Manual control disabled)" << std::endl;
                        } else {
                            std::cout << "No confirmed person to lock onto." << std::endl;
                        }
                    } else {
                        std::cout << "Robot must be standing to lock." << std::endl;
                    }
                } else if (key == 'u' || key == 'U') {
                    locked_person_id = -1;
                    reid_tracker.setLockedTrackId(-1);
                    sport_client.StopMove();
                    std::cout << "Unlocked." << std::endl;
                } else if (key == 'r' || key == 'R') {
                    locked_person_id = -1;
                    reid_tracker = ReIDTracker();
                    sport_client.StopMove();
                    std::cout << "Tracker reset." << std::endl;
                } else if (key == 'm' || key == 'M') {
                    bool current_mode = manual_control_active.load();
                    manual_control_active.store(!current_mode);
                    if (!current_mode) {
                        locked_person_id = -1;
                        reid_tracker.setLockedTrackId(-1);
                        sport_client.StopMove();
                        manual_moving = false;
                        std::cout << "MANUAL CONTROL MODE ENABLED (Use WASD + Space on video window)" << std::endl;
                        current_person_follow_status_global = "MANUAL CONTROL MODE";
                    } else {
                        sport_client.StopMove();
                        manual_moving = false;
                        std::cout << "MANUAL CONTROL MODE DISABLED" << std::endl;
                        current_person_follow_status_global = "Press 'L' to lock person";
                    }
                } else if (key == 'j' || key == 'J') {
                    if (current_robot_physical_state.load() == SITTING) {
                        performStandUp(sport_client);
                        waiting_for_trigger = false;
                    } else if (current_robot_physical_state.load() == IDLE_STANDING || 
                               current_robot_physical_state.load() == FOLLOWING) {
                        performSit(sport_client, false);
                        locked_person_id = -1;
                        waiting_for_trigger = true;
                    }
                } else if (manual_control_active.load()) {
                    // Handle WASD in manual mode
                    switch (key) {
                        case 'w':
                        case 'W':
                            manual_vx = MANUAL_MOVE_SPEED;
                            manual_vyaw = 0.0f;
                            manual_moving = true;
                            sport_client.Move(manual_vx, 0, manual_vyaw);
                            std::cout << "Manual: Forward" << std::endl;
                            break;
                        case 's':
                        case 'S':
                            manual_vx = -MANUAL_MOVE_SPEED;
                            manual_vyaw = 0.0f;
                            manual_moving = true;
                            sport_client.Move(manual_vx, 0, manual_vyaw);
                            std::cout << "Manual: Backward" << std::endl;
                            break;
                        case 'a':
                        case 'A':
                            manual_vx = 0.0f;
                            manual_vyaw = MANUAL_TURN_SPEED;
                            manual_moving = true;
                            sport_client.Move(manual_vx, 0, manual_vyaw);
                            std::cout << "Manual: Rotate Left" << std::endl;
                            break;
                        case 'd':
                        case 'D':
                            manual_vx = 0.0f;
                            manual_vyaw = -MANUAL_TURN_SPEED;
                            manual_moving = true;
                            sport_client.Move(manual_vx, 0, manual_vyaw);
                            std::cout << "Manual: Rotate Right" << std::endl;
                            break;
                        case ' ':
                            manual_vx = manual_vyaw = 0.0f;
                            manual_moving = false;
                            sport_client.StopMove();
                            std::cout << "Manual: Stop" << std::endl;
                            break;
                    }
                }
                
                // Send follow commands (unless in manual mode or waiting)
                if (!waiting_for_trigger && current_robot_physical_state.load() != SITTING) {
                    sendRobotCommand(sport_client, frame.cols, frame.rows);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    cv::destroyAllWindows();
    sport_client.StopMove();
    metrics_finalize();
    return 0;
}
