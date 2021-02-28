#include "extractor.h"

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


void Extractor::EstimateMotion(std::vector<cv::Point2f>& keypoints_1,
                               std::vector<cv::Point2f>& keypoints_2,
                               const cv::Mat& K, cv::Mat& R, cv::Mat& t) {
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(keypoints_1, keypoints_2, K, cv::RANSAC, 0.99, 1.0, inlier_mask);
    cv::recoverPose(E, keypoints_1, keypoints_2, K, R, t, inlier_mask);              
    
    // TODO: filter outliers         
}

void Extractor::EstimateMotion(const std::vector<cv::Point2f>& keypoints,
                               const std::vector<cv::Vec3f>& landmarks,
                               cv::Mat& R, cv::Mat& t) {

}

void Extractor::FilterKeypoints(std::vector<cv::KeyPoint>& keypoints_1,
                                std::vector<cv::KeyPoint>& keypoints_2,
                                std::vector<std::vector<cv::DMatch>>& matches,
                                std::vector<cv::Point2f>& keypoints_1_m,
                                std::vector<cv::Point2f>& keypoints_2_m,
                                std::vector<cv::Point2f>& keypoints_1_nm,
                                std::vector<cv::Point2f>& keypoints_2_nm) {
    keypoints_1_m.clear();
    keypoints_2_m.clear();
    keypoints_1_nm.clear();
    keypoints_2_nm.clear();

    // Determine used and un-used indices
    std::vector<int> indices;
    push_back(indices, boost::irange(0, static_cast<int>(keypoints_1.size())));
    std::set<int> indices_set(indices.begin(), indices.end());

    std::set<int> indices_set_1_m, indices_set_2_m;
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        int query_idx = (*it)[0].queryIdx;
        int train_idx = (*it)[0].trainIdx;
        indices_set_1_m.insert(query_idx);
        indices_set_2_m.insert(train_idx);
        keypoints_1_m.push_back(keypoints_1[query_idx].pt);
        keypoints_2_m.push_back(keypoints_2[train_idx].pt);
    }

    std::set<int> indices_set_1_nm, indices_set_2_nm;
    std::set_difference(indices_set.begin(), indices_set.end(),
                        indices_set_1_m.begin(), indices_set_1_m.end(),
                        std::inserter(indices_set_1_nm, indices_set_1_nm.end()));
    std::set_difference(indices_set.begin(), indices_set.end(),
                        indices_set_2_m.begin(), indices_set_2_m.end(),
                        std::inserter(indices_set_2_nm, indices_set_2_nm.end()));
    
    // Generate unused keypoint vectors
    for (auto it = indices_set_1_nm.begin(); it != indices_set_1_nm.end(); ++it) {
        keypoints_1_nm.push_back(keypoints_1[*it].pt);
    }
    for (auto it = indices_set_2_nm.begin(); it != indices_set_2_nm.end(); ++it) {
        keypoints_2_nm.push_back(keypoints_2[*it].pt);
    }
}