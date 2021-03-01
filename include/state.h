#ifndef STATE_H_
#define STATE_H_

#include <vector>

#include <opencv2/core.hpp>

struct State {
    std::vector<cv::Point3f> trajectory;
    std::vector<cv::Point3f> landmarks;
    std::vector<cv::Point2f> kps_candidate;
    std::vector<cv::Point2f> kps;
    int step = 0;
};

#endif  // STATE_H_
