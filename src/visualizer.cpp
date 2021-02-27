#include "visualizer.h"

Visualizer::Visualizer(std::shared_ptr<State> state) {
    state_ = state;
    // plot_ = std::make_unique<plot::Plot>("Visualization");
}

void Visualizer::UpdateRender(std::shared_ptr<cv::Mat> current_frame) {
    // TODO: Draw state on frame
    current_vis_ = current_frame->clone();

    std::vector<float> x, y;
    x.push_back(0); y.push_back(0);

    plt::title("Visualization");
    plt::subplot(2, 2, 1);
    plt::imshow(current_vis_.data, current_vis_.rows, current_vis_.cols, 3);
    plt::subplot(2, 4, 5);
    plt::plot(x, y);
    plt::axis("equal");
    plt::subplot(2, 4, 6);
    plt::plot(x, y);
    plt::axis("equal");
    plt::subplot(1, 2, 2);
    plt::plot(x, y);
    plt::axis("equal");
    plt::show(false);
    plt::pause(0.001);
    // cv::imshow("Visualizer", current_vis_);
    // cv::waitKey(1);
}
