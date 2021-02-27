#ifndef VISUALIZER_H_
#define VISUALIZER_H_

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "matplotlibcpp.h"

#include "state.h"

namespace plt = matplotlibcpp;

class Visualizer {
 public:
    Visualizer(std::shared_ptr<State> state);
    void UpdateRender(std::shared_ptr<cv::Mat> current_frame);
 private:
    cv::Mat current_vis_;
    std::shared_ptr<State> state_;
    std::unique_ptr<plt::Plot> plot_;
};

#endif  // VISUALIZER_H_
