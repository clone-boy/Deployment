#ifndef ARCFACE_H
#define ARCFACE_H

#include"application/trt_infer.h"
#include"application/tensor.h"
#include<string>
#include<memory>

class Arcface{
public:
    Arcface(int bs=1):batch_size_(bs){}
    ~Arcface(){
        arcInfer_.reset();
    }

    void inference();
    void create_infer(const std::string& engine_file, int gpuid=0);

    void set_input(std::vector<img_bin> img_data, int index=0);
    std::shared_ptr<Tensor> get_output(int index=0);

public:
    int batch_size_;
    std::shared_ptr<Infer> arcInfer_ = nullptr;
};

#endif
