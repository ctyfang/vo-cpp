#include "pipeline.h"


Pipeline::Pipeline(const cv::FileStorage& param_node) {
    if (std::string(param_node["source"]) == "kitti") {
        dataloader_ = std::make_shared<KittiLoader>(std::string(param_node["root_dir"]), param_node["buffer_size"]);
    }

    extractor_ = std::make_unique<Extractor>();
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
    extractor_->EstimateMotion(kp_1_m, kp_2_m, dataloader_->K_, R, t);
    std::cout << R << "\n";
    std::cout << t << "\n";
 
    // Landmark creation

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