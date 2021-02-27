#include "pipeline.h"


Pipeline::Pipeline(const cv::FileStorage& param_node) {
    if (std::string(param_node["source"]) == "kitti") {
        dataloader_ = std::make_shared<KittiLoader>(std::string(param_node["root_dir"]), param_node["buffer_size"]);
    }

    visualizer_ = std::make_unique<Visualizer>(state_);
}

Pipeline::~Pipeline() {}

void Pipeline::Initialize() {
    // TODO: Bootstrap implementation
}

void Pipeline::Update() {
    std::shared_ptr<cv::Mat> img_ptr;
    bool img_ret = dataloader_->Read(img_ptr);
    if (img_ret) {
        // TODO: Processing
        visualizer_->UpdateRender(img_ptr);
    }
}