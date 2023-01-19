#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include<cuda_runtime.h>
#include<vector>
#include<string>

#define checkCudaRuntime(op) CUDA_TOOLS::check_cuda_runtime((op), #op, __FILE__, __LINE__)


namespace CUDA_TOOLS{
    bool check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);
    bool check_device_id(int device_id);
    std::vector<unsigned char> load_file(const std::string& file);

    class AutoDevice{
    public:
        AutoDevice(int device_id = 0);
        ~AutoDevice();
    
    private:
        int old_ = -1;
    };

}// namespace

#endif