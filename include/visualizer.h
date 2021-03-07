#ifndef VISUALIZER_H_
#define VISUALIZER_H_

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "matplotlibcpp.h"

#include "state.h"

namespace plt = matplotlibcpp;

class Visualizer {
 public:
    Visualizer(std::shared_ptr<State> state);
    void UpdateRender(std::shared_ptr<cv::Mat> current_frame);
 private:
   void DrawKeyPoints();

    cv::Mat current_vis_;
    std::vector<int> landmark_length_history_;
    std::shared_ptr<State> state_;
};

#endif  // VISUALIZER_H_
