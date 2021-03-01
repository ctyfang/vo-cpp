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
#include <opencv2/video/tracking.hpp>

class Extractor {
 public:
    Extractor(const cv::Mat& K);
    void ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors);
    void ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors);
    void MatchSIFT(const cv::Mat& descriptors_1, const cv::Mat& descriptors_2, 
                   std::vector<std::vector<cv::DMatch>>& matches, const float ratio_threshold = 0.8);
    void EstimateMotion(std::vector<cv::Point2f>& keypoints_1,
                        std::vector<cv::Point2f>& keypoints_2,
                        cv::Mat& R, cv::Mat& t); 
    void EstimateMotion(const std::vector<cv::Point2f>& keypoints,
                        const std::vector<cv::Point3f>& landmarks,
                        cv::Mat& R, cv::Mat& t);
    void FilterKeypoints(std::vector<cv::KeyPoint>& keypoints_1,
                         std::vector<cv::KeyPoint>& keypoints_2,
                         std::vector<std::vector<cv::DMatch>>& matches,
                         std::vector<cv::Point2f>& keypoints_1_m,
                         std::vector<cv::Point2f>& keypoints_2_m,
                         std::vector<cv::Point2f>& keypoints_1_nm,
                         std::vector<cv::Point2f>& keypoints_2_nm);
   cv::Point3f ComputePosition(cv::Mat& R, cv::Mat& t);
   std::vector<cv::Point3f> Triangulate(std::vector<cv::Point2f> keypoints_1,
                                        std::vector<cv::Point2f> keypoints_2,
                                        cv::Mat& R1, cv::Mat& t1,
                                        cv::Mat& R2, cv::Mat& t2);
   std::vector<uchar> TrackKeypoints(std::shared_ptr<cv::Mat> frame_curr, 
                                     std::vector<cv::Point2f>& keypoints);  
   void FilterKeypointsAndLandmarks(std::vector<cv::Point2f>& keypoints,
                                    std::vector<cv::Point3f>& landmarks,
                                    std::vector<uchar>& mask);
 private:
   cv::Mat K_;
   cv::Mat frame_prev_;
   cv::Ptr<cv::SIFT> sift_;
   cv::Ptr<cv::DescriptorMatcher> matcher_;
};

#endif  // EXTRACTOR_H_
