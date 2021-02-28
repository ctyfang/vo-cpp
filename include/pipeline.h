#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <memory>
#include <vector>

#include "state.h"
#include "dataloader.h"
#include "extractor.h"
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
    std::unique_ptr<Extractor> extractor_;
    std::unique_ptr<Visualizer> visualizer_;
    int frame_index_ = 0;
};

#endif  // PIPELINE_H_
