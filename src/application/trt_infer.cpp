#include"application/trt_infer.h"
#include"net/logging.h"

DataType convert_trt_datatype(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT: return DataType::Float;
        //case nvinfer1::DataType::kHALF: return DataType::Float16;
        default:
            error("Unsupport data type %d", dt);
            return DataType::Float;
    }
}


bool EngineContext::build_model(const void* pdata, size_t size){
    destroy();

    if(pdata == nullptr || size == 0)
        return false;

    checkCudaRuntime(cudaStreamCreate(&stream_));
    if(stream_ == nullptr)
        return false;

    runtime_ = make_nvshared(nvinfer1::createInferRuntime(logger_));
    if (runtime_ == nullptr)
        return false;

    engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size));
    if (engine_ == nullptr)
        return false;

    context_ = make_nvshared(engine_->createExecutionContext());
    return context_ != nullptr;
}

void EngineContext::destroy(){
    context_.reset();
    engine_.reset();
    runtime_.reset();

    if (stream_) 
        cudaStreamDestroy(stream_);
    stream_ = nullptr;
}


Infer::~Infer(){
    destroy();
}

void Infer::destroy() {
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_id_));
    context_.reset();
    outputs_.clear();
    inputs_.clear();
    inputs_name_.clear();
    outputs_name_.clear();
    orderdBlobs_.clear();
    bindingsPtr_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));
}

bool Infer::load(const std::string& file){
    auto data = CUDA_TOOLS::load_file(file);
    if (data.empty())
        return false;

    context_.reset(new EngineContext());

    //build model
    if (!context_->build_model(data.data(), data.size())) {
        context_.reset();
        return false;
    }

    return true;
}

void Infer::init(int batch_size){
    int nbBindings = context_->engine_->getNbIOTensors();

    inputs_.clear();
    inputs_name_.clear();
    outputs_.clear();
    outputs_name_.clear();
    orderdBlobs_.clear();
    bindingsPtr_.clear();

    for (int i = 0; i < nbBindings; ++i) {
        const char* bindingName = context_->engine_->getIOTensorName(i);
        auto dims = context_->engine_->getTensorShape(bindingName);
        auto type = context_->engine_->getTensorDataType(bindingName);
        dims.d[0] = batch_size;
        auto newTensor = std::make_shared<Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type), device_id_);
        newTensor->set_stream(context_->stream_);
        if (context_->engine_->getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT) {
            //input
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
        } else {
            //output
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
        }
		orderdBlobs_.push_back(newTensor);
    }
    bindingsPtr_.resize(orderdBlobs_.size());
}

std::shared_ptr<Tensor> Infer::input(int index){
    if(index < 0 || index >= inputs_.size()){
        fatal("Input index[%d] out of range [size=%d]", index, inputs_.size());
    }
    return inputs_[index];
}

std::shared_ptr<Tensor> Infer::output(int index){
    if(index < 0 || index >= outputs_.size()){
        fatal("Output index[%d] out of range [size=%d]", index, outputs_.size());
    }
    return outputs_[index];
}

void Infer::forward1(bool sync){
    EngineContext* context = (EngineContext*)context_.get();
    for (int i = 0; i < orderdBlobs_.size(); ++i) {
        orderdBlobs_[i]->to_gpu();
    }

    int inputBatchSize = inputs_[0]->size(0);
    for(int i = 0; i < context->engine_->getNbIOTensors(); ++i){
        const char* bindingName = context->engine_->getIOTensorName(i);
        auto dims = context->engine_->getTensorShape(bindingName);
        dims.d[0] = inputBatchSize;
        if(context->engine_->getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT){
            context->context_->setInputShape(bindingName, dims);
        }
    }

    for (int i = 0; i < orderdBlobs_.size(); ++i)
        bindingsPtr_[i] = orderdBlobs_[i]->gpu();

    void** bindingsptr = bindingsPtr_.data();
    bool execute_result = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);  // Deprecated in TensorRT 8.5
    if(!execute_result){
        auto code = cudaGetLastError();
        fatal("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
    }

    if (sync) {
        synchronize();
    }
}

void Infer::forward2(bool sync){
    EngineContext* context = (EngineContext*)context_.get();
    for (int i = 0; i < orderdBlobs_.size(); ++i) {
        orderdBlobs_[i]->to_gpu();
    }

    int inputBatchSize = orderdBlobs_[0]->size(0);
    for(int i = 0; i < context->engine_->getNbIOTensors(); ++i){
        const char* bindingName = context->engine_->getIOTensorName(i);
        if(context->engine_->getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT){
            auto dims = context->engine_->getTensorShape(bindingName);
            dims.d[0] = inputBatchSize;
            context->context_->setInputShape(bindingName, dims);
            context->context_->setTensorAddress(bindingName, orderdBlobs_[i]->gpu());
        } else{
            context->context_->setTensorAddress(bindingName, orderdBlobs_[i]->gpu());
        }
    }

    bool execute_result = context->context_->enqueueV3(context->stream_);
    if(!execute_result){
        auto code = cudaGetLastError();
        fatal("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
    }

    if (sync) {
        synchronize();
    }
}

void Infer::synchronize() {
    checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
}