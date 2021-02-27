#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>

#include "state.h"
#include "dataloader.h"
#include "visualizer.h"

class Pipeline {
 public:
    Pipeline(const cv::FileStorage& param_node);
    ~Pipeline();
    void Initialize();
    void Update();

 private:
    std::shared_ptr<State> state_;
    std::shared_ptr<Dataloader> dataloader_;
    std::unique_ptr<Visualizer> visualizer_;
};

#endif  // PIPELINE_H_
