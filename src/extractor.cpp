#include "extractor.h"
#include <iostream>

Extractor::Extractor(const cv::Mat& K)
{
    K_ = K.clone();
    sift_ = cv::SIFT::create();
    matcher_ = cv::DescriptorMatcher::create("BruteForce");
}

void Extractor::ExtractSIFT(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors) {
    sift_->detectAndCompute(*frame, cv::Mat(), keypoints, descriptors);
    frame_prev_ = frame->clone();
}

void Extractor::ExtractCorners(std::shared_ptr<cv::Mat> frame, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    frame_prev_ = frame->clone();
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
                               cv::Mat& R, cv::Mat& t) {
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(keypoints_1, keypoints_2, K_, cv::RANSAC, 0.99, 1.0, inlier_mask);
    cv::recoverPose(E, keypoints_1, keypoints_2, K_, R, t, inlier_mask);

    int lambda_idx = 0;
    keypoints_1.erase(std::remove_if(keypoints_1.begin(), keypoints_1.end(),
                                     [&lambda_idx, inlier_mask](cv::Point2f& kp) {
                                         return inlier_mask.at<bool>(lambda_idx++, 0) == 0;
                                     }), keypoints_1.end());
    lambda_idx = 0;
    keypoints_2.erase(std::remove_if(keypoints_2.begin(), keypoints_2.end(),
                                     [&lambda_idx, inlier_mask](cv::Point2f& kp) {
                                         return inlier_mask.at<bool>(lambda_idx++, 0) == 0;
                                     }), keypoints_2.end());
}

void Extractor::EstimateMotion(const std::vector<cv::Point2f>& keypoints,
                               const std::vector<cv::Point3f>& landmarks,
                               cv::Mat& R, cv::Mat& t) {
    cv::Mat inlier_mask;
    cv::solvePnPRansac(landmarks, keypoints, K_, cv::Mat(), R, t, false, 1000, 1.0, 0.99, inlier_mask);
    cv::Rodrigues(R, R);
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

cv::Point3f Extractor::ComputePosition(cv::Mat& R, cv::Mat& t) {
    cv::Mat pos_mat = R.inv() * (-t);
    pos_mat.convertTo(pos_mat, CV_32F);
    return cv::Point3f(pos_mat.at<float>(0, 0), pos_mat.at<float>(1, 0), pos_mat.at<float>(2, 0));
}

std::vector<cv::Point3f> Extractor::Triangulate(std::vector<cv::Point2f> keypoints_1,
                                                std::vector<cv::Point2f> keypoints_2,
                                                cv::Mat& R1, cv::Mat& t1,
                                                cv::Mat& R2, cv::Mat& t2) {
    // Generate projection matrices
    cv::Mat P1, P2;
    cv::hconcat(R1, t1, P1);
    cv::hconcat(R2, t2, P2);
    P1.convertTo(P1, CV_32F);
    P2.convertTo(P2, CV_32F);
    P1 = K_ * P1;
    P2 = K_ * P2;

    // Triangulate
    cv::Mat landmarks_homo(4, keypoints_1.size(), CV_32F);
    cv::triangulatePoints(P1, P2, keypoints_1, keypoints_2, landmarks_homo);
    
    // TODO: Non-linear refinement

    // De-homogenize
    std::vector<cv::Point3f> landmarks;
    for (int i = 0; i < keypoints_1.size(); ++i) {
        float w = landmarks_homo.at<float>(3, i);
        landmarks.push_back(cv::Point3f(landmarks_homo.at<float>(0, i)/w,
                                        landmarks_homo.at<float>(1, i)/w,
                                        landmarks_homo.at<float>(2, i)/w));
    }
    return landmarks;
}

std::vector<uchar> Extractor::TrackKeypoints(std::shared_ptr<cv::Mat> frame_curr,
                                             std::vector<cv::Point2f>& keypoints) {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(frame_prev_, *frame_curr,
                             keypoints, keypoints, status, err);
    
    // TODO: Bi-directional error checking
    frame_prev_ = frame_curr->clone();

    return status;
}

void Extractor::FilterKeypointsAndLandmarks(std::vector<cv::Point2f>& keypoints,
                                            std::vector<cv::Point3f>& landmarks,
                                            std::vector<uchar>& mask) {
    int lambda_idx = 0;
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),
                                   [mask, &lambda_idx](cv::Point2f kp){
                                       return mask[lambda_idx++] == 0;
                                   }), keypoints.end());
    lambda_idx = 0;
    landmarks.erase(std::remove_if(landmarks.begin(), landmarks.end(),
                                   [mask, &lambda_idx](cv::Point3f lm){
                                       return mask[lambda_idx++] == 0;
                                   }), landmarks.end());
}