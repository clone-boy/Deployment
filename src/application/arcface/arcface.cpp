#include"application/arcface/arcface.h"
#include<opencv2/opencv.hpp>
#include"net/logging.h"
#include"application/cuda_tools.h"

void Arcface::inference(){
    //arcInfer_->forward1(true);
    arcInfer_->forward2(true);
}

void Arcface::create_infer(const std::string& engine_file, int gpuid){
    arcInfer_ = std::make_shared<Infer>(gpuid);
    check(arcInfer_->load(engine_file), "load engine error");
    arcInfer_->init(batch_size_);
}

void Arcface::set_input(std::vector<img_bin> img_data, int index){
    std::shared_ptr<Tensor> tensor = arcInfer_->input(index);
    float mean[3]={0.5f,0.5f,0.5f};
    float std[3]={0.5f,0.5f,0.5f};
    for(int i=0;i<batch_size_;i++){
        cv::Mat img = cv::imdecode(img_data[i],cv::IMREAD_COLOR);
        tensor->from_norm_cv_img(i, img, mean, std);
    }
}

std::shared_ptr<Tensor> Arcface::get_output(int index){
    auto new_tensor = arcInfer_->output(index)->clone();
    //auto new_tensor = arcInfer_->output(index);
    return new_tensor;
}