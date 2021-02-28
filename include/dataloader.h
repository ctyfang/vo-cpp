#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <vector>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <thread>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std::filesystem;

class Dataloader {
 public:
    bool Read(std::shared_ptr<cv::Mat>& img_ptr);
 protected:
    virtual void UpdateBuffer() = 0;
    virtual void ParseCalibration() = 0;

    int buffer_size_;
    std::vector<std::shared_ptr<cv::Mat>> buffer_;
    std::unique_ptr<std::thread> buffer_thread_;
    std::mutex buffer_lock_;
};

class KittiLoader : public virtual Dataloader {
 public:
    KittiLoader(std::string root_dir, int buffer_size = 10);
    ~KittiLoader();
 private:
    void UpdateBuffer();
    void ParseCalibration();

    std::string image_dir_;
    int image_index_ = 0;
    cv::Mat K_;
};

#endif  // DATALOADER_H_
