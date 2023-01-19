#include"application/tensor.h"
#include<cuda_fp16.h>
#include"net/logging.h"
#include"application/cuda_tools.h"

float float16_to_float(float16 value){
		return __half2float(*reinterpret_cast<__half*>(&value));
}

float16 float_to_float16(float value){
    auto val = __float2half(value);
    return *reinterpret_cast<float16*>(&val);
}

int data_type_size(DataType dt){
    switch (dt) {
        case DataType::Float: return sizeof(float);
        //case DataType::Float16: return sizeof(float16);
        default: {
            error("Not support dtype: %d", dt);
            return -1;
        }
    }
}

const char* data_head_string(DataHead dh){
    switch(dh){
        case DataHead::Init: return "Init";
        case DataHead::Host: return "Host";
        case DataHead::Device: return "Device";
        default: return "Unknow";
    }
}

const char* data_type_string(DataType dt){
    switch(dt){
        case DataType::Float: return "Float32";
        //case DataType::Float16: return "Float16";
        default: return "Unknow";
    }
}

inline static int get_device(int device_id){
    bool con = true;
    if(device_id != HOST_ID){
        con = CUDA_TOOLS::check_device_id(device_id);
    }

    if(con)
        return device_id;
    else
        return HOST_ID;
}

MixMemory::MixMemory(int device_id){
    device_id_ = get_device(device_id);
}

MixMemory::~MixMemory() {
    release();
}

void MixMemory::malloc(size_t size) {
    data_size_ = size;
    CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);
    checkCudaRuntime(cudaMalloc(&gpu_, size));
    checkCudaRuntime(cudaMemset(gpu_, 0, size));

    checkCudaRuntime(cudaMallocHost(&cpu_, size));
    check(cpu_ != nullptr, "malloc host error");
    memset(cpu_, 0, size);
}

void* MixMemory::to_gpu(){
    if(cpu_){
        CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);
        checkCudaRuntime(cudaMemcpyAsync(gpu_,cpu_, data_size_, cudaMemcpyHostToDevice, stream_));
    }
    return gpu_;
}
void* MixMemory::to_cpu(){
    if(gpu_){
        CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);
        checkCudaRuntime(cudaMemcpyAsync(cpu_, gpu_, data_size_, cudaMemcpyDeviceToHost, stream_));
        checkCudaRuntime(cudaStreamSynchronize(stream_));
    }
    return cpu_;
}

void MixMemory::clone_cpu(std::shared_ptr<MixMemory> data){
    check(this->data_size()==data->data_size(), "clone data size error");
    memcpy(this->cpu(), data->cpu(), data_size_);
}

void MixMemory::clone_gpu(std::shared_ptr<MixMemory> data){
    check(this->data_size()==data->data_size(), "clone data size error");
    CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);

    checkCudaRuntime(cudaMemcpy(this->gpu(), data->gpu(), data_size_, cudaMemcpyDeviceToDevice));
}

void MixMemory::release_cpu() {
    if (cpu_) {
        CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);
        checkCudaRuntime(cudaFreeHost(cpu_));
        cpu_ = nullptr;
    }
}

void MixMemory::release_gpu() {
    if (gpu_) {
        CUDA_TOOLS::AutoDevice auto_device_exchange(device_id_);
        checkCudaRuntime(cudaFree(gpu_));
        gpu_ = nullptr;
    }
}

void MixMemory::release() {
    release_cpu();
    release_gpu();
    data_size_ = 0;
    stream_ = nullptr;
}



Tensor::Tensor(const std::vector<int>& dims, DataType dtype, int device_id){
    dtype_ = dtype;
    device_id_ = device_id;
    set_shape(dims);
    setup_data();
}

Tensor::Tensor(int ndims, const int* dims, DataType dtype, int device_id) {
    dtype_ = dtype;
    device_id_ = device_id;
    set_shape(ndims, dims);
    setup_data();
}

void Tensor::set_shape(int ndims, const int* dims){
    std::vector<int> setup_dims(ndims);
    for(int i = 0; i < ndims; ++i){
        int dim = dims[i];
        check(dim>0, "input dim error");
        setup_dims[i] = dim;
    }
    shape_ = setup_dims;
}

void Tensor::setup_data(){
    data_ = std::make_shared<MixMemory>(device_id_);
    device_id_ = data_->device_id();
    
    if(head_ == DataHead::Init && device_id_ == HOST_ID){
        head_ = DataHead::Host;
    } else if(head_ == DataHead::Init && device_id_ != HOST_ID){
        head_ = DataHead::Device;
    }
    bytes_ = numel() * element_size();
    data_->malloc(bytes_);
}

Tensor::~Tensor() {
    release();
}

Tensor& Tensor::release() {
    data_.reset();
    shape_.clear();
    bytes_ = 0;
    head_ = DataHead::Init;
    stream_ = nullptr;
    return *this;
}

int Tensor::numel() const{
    int value = shape_.empty() ? 0 : 1;
    for(int i = 0; i < shape_.size(); ++i){
        value *= shape_[i];
    }
    return value;
}

int Tensor::count(int start_axis) const {

    if(start_axis >= 0 && start_axis < shape_.size()){
        int size = 1;
        for (int i = start_axis; i < shape_.size(); ++i) 
            size *= shape_[i];
        return size;
    }else{
        return 0;
    }
}

std::shared_ptr<Tensor> Tensor::clone() const{
    auto new_tensor = std::make_shared<Tensor>(shape_, dtype_, device_id_);
    
    if(head_ == DataHead::Host){
        new_tensor->data_->clone_cpu(this->get_data());
    }else if(head_ == DataHead::Device){
        new_tensor->data_->clone_gpu(this->get_data());
    }
    
    new_tensor->head_ = this->head();

    return new_tensor;
}

bool Tensor::empty() const{
    return bytes_ == 0;
}


int Tensor::offset_array(size_t size, const int* index_array) const{
    check(size <= shape_.size(), "index range error");
    int value = 0;
    for(int i = 0; i < shape_.size(); ++i){
        if(i < size)
            value += index_array[i];

        if(i + 1 < shape_.size())
            value *= shape_[i+1];
    }
    return value;
}

int Tensor::offset_array(const std::vector<int>& index_array) const{
    return offset_array(index_array.size(), index_array.data());
}

Tensor& Tensor::to_gpu() {

    if (head_ == DataHead::Device)
        return *this;

    head_ = DataHead::Device;
    data_->to_gpu();
    return *this;
}

Tensor& Tensor::to_cpu() {

    if (head_ == DataHead::Host)
        return *this;

    head_ = DataHead::Host;
    data_->to_cpu();
    return *this;
}


Tensor& Tensor::from_cv_img(int n, const cv::Mat& image){
    check(image.channels() == 3 && !image.empty(), "image error");
    check(ndims() == 4 && n < shape_[0], "dim error");
    to_cpu();

    int width   = shape_[3];
    int height  = shape_[2];
    cv::Mat inputframe;
    cv::cvtColor(image, inputframe, cv::COLOR_BGR2RGB);
    if(inputframe.size() != cv::Size(width, height))
        cv::resize(inputframe, inputframe, cv::Size(width, height));
    
    int image_area = width*height;
    unsigned char* pimage = inputframe.data;
	float* phost_r = cpu<float>(n, 0);
	float* phost_g = cpu<float>(n, 1);
	float* phost_b = cpu<float>(n, 2);
	for (int i=0; i<image_area; ++i, pimage += 3) {
        *phost_r++ = pimage[0];
        *phost_g++ = pimage[1];
        *phost_b++ = pimage[2];
	}
    to_gpu();
    return *this;
}

Tensor& Tensor::from_norm_cv_img(int n, const cv::Mat& image, float mean[3], float std[3]) {
    check(image.channels() == 3 && !image.empty(), "image error");
    check(ndims() == 4 && n < shape_[0], "dim error");
    to_cpu();

    int width  = shape_[3];
    int height = shape_[2];
    cv::Mat inputframe;
    cv::cvtColor(image, inputframe, cv::COLOR_BGR2RGB);
    if(inputframe.size() != cv::Size(width, height))
        cv::resize(inputframe, inputframe, cv::Size(width, height));
    
    int image_area = width*height;
    unsigned char* pimage = inputframe.data;
	float* phost_r = cpu<float>(n, 0);
	float* phost_g = cpu<float>(n, 1);
	float* phost_b = cpu<float>(n, 2);
	for (int i=0; i<image_area; ++i, pimage += 3) {
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
	}
    to_gpu();
    return *this;
}


bool Tensor::save_to_file(const std::string& file) const{

    if(empty()) return false;

    FILE* f = fopen(file.c_str(), "wb");
    if(f == nullptr) return false;

    int ndims = this->ndims();
    unsigned int head[3] = {0xFCCFE2E2, ndims, static_cast<unsigned int>(dtype_)};
    fwrite(head, 1, sizeof(head), f);
    fwrite(shape_.data(), 1, sizeof(shape_[0]) * shape_.size(), f);
    ((Tensor*)this)->to_cpu();
    fwrite(cpu(), 1, bytes_, f);
    fclose(f);
    return true;
}

std::shared_ptr<Tensor> load_from_file(const std::string& file, int device_id){

    FILE* f = fopen(file.c_str(), "rb");
    if(f == nullptr){
        error("Open %s failed.", file.c_str());
        return nullptr;
    }

    unsigned int head[3] = {0};
    fread(head, 1, sizeof(head), f);

    if(head[0] != 0xFCCFE2E2){
        fclose(f);
        error("Invalid tensor file %s, magic number mismatch", file.c_str());
        return nullptr;
    }

    int ndims = head[1];
    auto dtype = (DataType)head[2];
    std::vector<int> dims(ndims);
    fread(dims.data(), 1, ndims * sizeof(dims[0]), f);

    auto new_tensor = std::make_shared<Tensor>(dims, dtype, device_id);
    new_tensor->to_cpu();

    fread(new_tensor->cpu(), 1, new_tensor->bytes(), f);
    fclose(f);
    return new_tensor;
}