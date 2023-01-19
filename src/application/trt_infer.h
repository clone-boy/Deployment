#ifndef TRT_INFER_H
#define TRT_INFER_H

#include<NvInfer.h>
#include<memory>
#include"application/cuda_tools.h"
#include<NvInferRuntime.h>
#include"application/tensor.h"
#include"net/logging.h"

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if (severity == Severity::kINTERNAL_ERROR) {
            fatal("NVInfer INTERNAL_ERROR: %s", msg);
        }else if (severity == Severity::kERROR) {
            error("NVInfer ERROR: %s", msg);
        }
        else  if (severity == Severity::kWARNING) {
            warn("NVInfer WARNING: %s", msg);
        }
        else  if (severity == Severity::kINFO) {
            info("NVInfer INFO: %s", msg);
        }
        else {
            debug("NVInfer: %s", msg);
        }
    }
};

template<typename _T>
std::shared_ptr<_T> make_nvshared(_T *ptr) {
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

class EngineContext {
public:
    EngineContext(){}
    ~EngineContext() { destroy(); }

    bool build_model(const void* pdata, size_t size);

    void destroy();

public:
    TRTLogger logger_;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

DataType convert_trt_datatype(nvinfer1::DataType dt);

class Infer{
public:
    Infer(int device_id=0):device_id_(device_id){}
    ~Infer();
    void destroy();

    bool load(const std::string& file);
    void init(int batch_size);
    std::shared_ptr<Tensor> input (int index = 0);
    std::shared_ptr<Tensor> output(int index = 0);
    void forward1(bool sync);
    void forward2(bool sync);
    void synchronize();
public:
    int device_id_;
    std::shared_ptr<EngineContext> context_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
	std::vector<std::shared_ptr<Tensor>> outputs_;
    std::vector<std::string> inputs_name_;
	std::vector<std::string> outputs_name_;
    std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
    std::vector<void*> bindingsPtr_;
};



#endif