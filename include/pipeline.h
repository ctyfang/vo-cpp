#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>

#include "state.h"
#include "dataloader.h"

class Pipeline {
 public:
    Pipeline(const cv::FileStorage& param_node);
    ~Pipeline();

 private:
    State state_;
    std::shared_ptr<Dataloader> dataloader_;
};

#endif  // PIPELINE_H_
