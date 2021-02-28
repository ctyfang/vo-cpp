#ifndef EXTRACTOR_H_
#define EXTRACTOR_H_

#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

class Extractor {
 public:
    Extractor();
    void ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors);
    void ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors);
    void MatchSIFT(const cv::Mat& descriptors_1, const cv::Mat& descriptors_2, 
                   std::vector<std::vector<cv::DMatch>>& matches, const float ratio_threshold = 0.5);

 private:
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

#endif  // EXTRACTOR_H_
