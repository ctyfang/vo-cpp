#include "pipeline.h"


Pipeline::Pipeline(const cv::FileStorage& param_node) {
    if (std::string(param_node["source"]) == "kitti") {
        dataloader_ = std::make_shared<KittiLoader>(std::string(param_node["root_dir"]), param_node["buffer_size"]);
    }
}

Pipeline::~Pipeline() {
    
}
