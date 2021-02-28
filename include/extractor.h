#ifndef EXTRACTOR_H_
#define EXTRACTOR_H_

#include <vector>
#include <set>
#include <algorithm>

#include <boost/range/algorithm/set_algorithm.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

class Extractor {
 public:
    Extractor();
    void ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors);
    void ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors);
    void MatchSIFT(const cv::Mat& descriptors_1, const cv::Mat& descriptors_2, 
                   std::vector<std::vector<cv::DMatch>>& matches, const float ratio_threshold = 0.5);
    void EstimateMotion(std::vector<cv::Point2f>& keypoints_1,
                        std::vector<cv::Point2f>& keypoints_2,
                        const cv::Mat& K, cv::Mat& R, cv::Mat& t); 
    void EstimateMotion(const std::vector<cv::Point2f>& keypoints,
                        const std::vector<cv::Vec3f>& landmarks,
                        cv::Mat& R, cv::Mat& t);
    void FilterKeypoints(std::vector<cv::KeyPoint>& keypoints_1,
                         std::vector<cv::KeyPoint>& keypoints_2,
                         std::vector<std::vector<cv::DMatch>>& matches,
                         std::vector<cv::Point2f>& keypoints_1_m,
                         std::vector<cv::Point2f>& keypoints_2_m,
                         std::vector<cv::Point2f>& keypoints_1_nm,
                         std::vector<cv::Point2f>& keypoints_2_nm);
 private:
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

#endif  // EXTRACTOR_H_
