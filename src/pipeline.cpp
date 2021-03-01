#include "pipeline.h"


Pipeline::Pipeline(const cv::FileStorage& param_node) {
    if (std::string(param_node["source"]) == "kitti") {
        dataloader_ = std::make_shared<KittiLoader>(std::string(param_node["root_dir"]), param_node["buffer_size"]);
    }

    state_ = std::make_shared<State>();
    extractor_ = std::make_unique<Extractor>(dataloader_->K_);
    visualizer_ = std::make_unique<Visualizer>(state_);
}

Pipeline::~Pipeline() {}

void Pipeline::Initialize() {
    // Grab initializing frames
    int index_1 = 0, index_2 = 10;
    std::shared_ptr<cv::Mat> frame_1, frame_2, frame_curr;

    while (frame_index_ <= index_2) {
        bool ret = dataloader_->Read(frame_curr);

        if (ret) {
            if (frame_index_ == index_1) {
                frame_1 = frame_curr;
            } else if (frame_index_ == index_2) {
                frame_2 = frame_curr;
            }
            frame_index_++;
        }
    }

    // Feature extraction and matching
    std::vector<cv::KeyPoint> cv_kp_1, cv_kp_2;
    cv::Mat desc_1, desc_2;
    extractor_->ExtractSIFT(frame_1, cv_kp_1, desc_1);
    extractor_->ExtractSIFT(frame_2, cv_kp_2, desc_2);

    std::vector<std::vector<cv::DMatch>> matches;
    extractor_->MatchSIFT(desc_1, desc_2, matches);

    // // DEBUG: Draw matches
    // cv::Mat im_kp_1, im_kp_2, im_kps;
    // cv::drawKeypoints(*frame_1, kp_1, im_kp_1);
    // cv::drawKeypoints(*frame_2, kp_2, im_kp_2);
    // cv::vconcat(im_kp_1, im_kp_2, im_kps);
    // cv::imshow("Keypoints", im_kps);
    // cv::waitKey(0);

    // cv::Mat im_matches;
    // cv::drawMatches(*frame_1, kp_1, *frame_2, kp_2, matches, im_matches);
    // cv::imshow("Matches", im_matches);
    // cv::waitKey(0);

    // Motion estimation
    std::vector<cv::Point2f> kp_1_m, kp_2_m, kp_1_nm, kp_2_nm;
    extractor_->FilterKeypoints(cv_kp_1, cv_kp_2, matches, kp_1_m, kp_2_m, kp_1_nm, kp_2_nm);

    cv::Mat R, t;
    extractor_->EstimateMotion(kp_1_m, kp_2_m, R, t);

    state_->kps.insert(state_->kps.end(), kp_1_m.begin(), kp_1_m.end());
    state_->kps_candidate.insert(state_->kps_candidate.end(), kp_2_nm.begin(), kp_2_nm.end());

    cv::Point3f pos = extractor_->ComputePosition(R, t);
    state_->trajectory.push_back(pos);

    // Landmark creation
    cv::Mat R0 = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t0 = cv::Mat::zeros(3, 1, CV_32F);
    std::vector<cv::Point3f> landmarks_new = extractor_->Triangulate(kp_1_m, kp_2_m,
                                                                     R0, t0, R, t);

    state_->landmarks.insert(state_->landmarks.begin(), landmarks_new.begin(), landmarks_new.end());
    // Visualize
    visualizer_->UpdateRender(frame_2);
}

void Pipeline::Update() {
    // TODO: Implement continuous update and remove waitKey
    cv::waitKey(0);

    std::shared_ptr<cv::Mat> frame_curr;
    bool ret = dataloader_->Read(frame_curr);
    if (ret) {
        visualizer_->UpdateRender(frame_curr);
    }
}