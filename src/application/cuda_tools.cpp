#include"application/cuda_tools.h"
#include"net/logging.h"
#include<fstream>

namespace CUDA_TOOLS{
    bool check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
        if (code != cudaSuccess) {
            const char* err_name = cudaGetErrorName(code);
            const char* err_message = cudaGetErrorString(code);
            error("cuda runtime error %s: %d  %s failed.\n  code = %s, message = %s", file, line, op, err_name, err_message);
            return false;
	    }
	    return true;
    }

    bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            error("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    std::vector<unsigned char> load_file(const std::string& file) {
        std::ifstream in(file, std::ios::in | std::ios::binary);
        if (!in.is_open()) return {};

        in.seekg(0, std::ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, std::ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }

    AutoDevice::AutoDevice(int device_id){
        checkCudaRuntime(cudaGetDevice(&old_));
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    AutoDevice::~AutoDevice(){
        checkCudaRuntime(cudaSetDevice(old_));
    }
}