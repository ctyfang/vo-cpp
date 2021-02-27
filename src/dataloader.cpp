#include "dataloader.h"
#include <chrono>

#include <opencv2/highgui.hpp>

bool Dataloader::Read(std::shared_ptr<cv::Mat>& img_ptr) {
    buffer_lock_.lock();
    if (buffer_.size() > 0) {
        img_ptr = buffer_.front();
        buffer_.erase(buffer_.begin());
        buffer_lock_.unlock();
        return true;
    } else {
        buffer_lock_.unlock();
        return false;
    }
}

KittiLoader::KittiLoader(std::string root_dir, int buffer_size) {
    buffer_size_ = buffer_size;
    image_dir_ = root_dir + "/00/image_0";
    std::cout << "Image Directory: " << std::string(image_dir_) << "\n";
    buffer_thread_ = std::make_unique<std::thread>(&KittiLoader::UpdateBuffer, this);
    ParseCalibration();
}

KittiLoader::~KittiLoader() {
    buffer_thread_->join();
}

void KittiLoader::UpdateBuffer() {
    while (true) {
        buffer_lock_.lock();
        if (buffer_.size() < buffer_size_) {
            std::string image_filename = std::to_string(image_index_);
            image_index_ += 1;
            image_filename = std::string(6-image_filename.length(), '0').append(image_filename).append(".png");
            buffer_.push_back(std::make_shared<cv::Mat>(cv::imread(image_dir_ + "/" + image_filename)));
        }
        buffer_lock_.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void KittiLoader::ParseCalibration() {
    // TODO: implement
}
