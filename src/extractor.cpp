#include "extractor.h"

#include <iostream>

Extractor::Extractor()
{
    sift_ = cv::SIFT::create();
    matcher_ = cv::DescriptorMatcher::create("BruteForce");
}

void Extractor::ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors) {
    sift_->detectAndCompute(*frame, cv::Mat(), keypoints, descriptors);
}

void Extractor::ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    
}

void Extractor::MatchSIFT(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
                          std::vector<std::vector<cv::DMatch>> &matches, const float ratio_threshold)
{
    matcher_->knnMatch(descriptors_1, descriptors_2, matches, 2);

    // Filter matches via ratio test
    matches.erase(std::remove_if(matches.begin(), matches.end(),
                                 [ratio_threshold](std::vector<cv::DMatch> m) {
                                     if (m[0].distance / m[1].distance > ratio_threshold) {
                                         return true;
                                     } else {
                                         return false;
                                     }
                                 }),
                  matches.end());
}
