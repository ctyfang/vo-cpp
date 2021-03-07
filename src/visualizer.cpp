#include "visualizer.h"

Visualizer::Visualizer(std::shared_ptr<State> state) {
    state_ = state;
}

void Visualizer::DrawKeyPoints() {
    for (cv::Point2f pt : state_->kps) {
        cv::circle(current_vis_, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
    for (cv::Point2f pt : state_->kps_candidate) {
        cv::circle(current_vis_, pt, 3, cv::Scalar(255, 0, 0), -1);
    }
}

void Visualizer::UpdateRender(std::shared_ptr<cv::Mat> current_frame) {
    // TODO: More efficient way to visualize...
    
    // Draw keypoints
    current_vis_ = current_frame->clone();
    DrawKeyPoints();

    // Extract trajectory data
    std::vector<float> pos_x, pos_z;
    for (cv::Point3f pos : state_->trajectory) {
        pos_x.push_back(pos.x);
        pos_z.push_back(pos.z);
    }

    // Extract landmark data
    std::vector<float> lm_x, lm_z;
    for (cv::Point3f lm : state_->landmarks) {
        lm_x.push_back(lm.x);
        lm_z.push_back(lm.z);
    }

    // Record number of landmarks
    landmark_length_history_.push_back(state_->landmarks.size());

    // Plots
    plt::title("Visualization");
    std::map<std::string, std::string> lm_settings, traj_settings;
    lm_settings.insert({"c", "red"});
    traj_settings.insert({"c", "green"});

    plt::subplot(2, 2, 1);
    plt::imshow(current_vis_.data, current_vis_.rows, current_vis_.cols, 3);

    plt::subplot(2, 4, 5);
    plt::plot(landmark_length_history_, lm_settings);

    plt::subplot(2, 4, 6);
    plt::scatter(lm_x, lm_z, 2.0, lm_settings);
    plt::scatter(pos_x, pos_z, 2.0, traj_settings);

    plt::subplot(1, 2, 2);
    plt::scatter(lm_x, lm_z, 2.0, lm_settings);

    plt::show(false);
    plt::pause(0.001);
}
