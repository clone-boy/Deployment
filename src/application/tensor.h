#ifndef Tensor_H
#define Tensor_H

#include<vector>
#include<string>
#include<memory>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>

typedef std::vector<uint8_t> img_bin;

const static int HOST_ID=-1;

typedef struct{unsigned short _;} float16;

enum class DataHead : int{
    Init   = 0,
    Host   = 1,
    Device = 2
};

enum class DataType : int {
    Unknow = -1,
    Float = 0,
    //Float16 = 1,
};

int data_type_size(DataType dt);
float float16_to_float(float16 value);
float16 float_to_float16(float value);
const char* data_head_string(DataHead dh);
const char* data_type_string(DataType dt);


class MixMemory {
public:
    MixMemory(int device_id = HOST_ID);
    ~MixMemory();
    void malloc(size_t size);
    void* to_gpu();
    void* to_cpu();
    void release_gpu();
    void release_cpu();
    void release();

    void set_stream(cudaStream_t stream){ stream_ = stream; }

    inline size_t data_size() const{return data_size_;}
    inline int device_id() const{return device_id_;}
    inline void set_device(int id) { device_id_ = id; }

    inline void* gpu() const { return gpu_; }

    // Pinned Memory
    inline void* cpu() const { return cpu_; }

    void clone_cpu(std::shared_ptr<MixMemory> data);
    void clone_gpu(std::shared_ptr<MixMemory> data);

private:
    void* cpu_ = nullptr;
    int device_id_ = HOST_ID;

    void* gpu_ = nullptr;
    size_t data_size_ = 0;

    cudaStream_t stream_;
};


class Tensor {
public:
    Tensor(const Tensor& other) = delete;
    Tensor& operator = (const Tensor& other) = delete;

    explicit Tensor(int ndims, const int* dims, DataType dtype = DataType::Float, int device_id = HOST_ID);
    explicit Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float, int device_id = HOST_ID);
    ~Tensor();

    int numel() const;
    int count(int start_axis = 0) const;
    inline int batch()   const{return shape_[0];}

    inline DataType dtype()                const { return dtype_; }
    inline const std::vector<int>& shape() const { return shape_; }
    inline const std::vector<int>& size()  const { return shape_; }
    inline int size(int index)  const{return shape_[index];}
    inline int shape(int index) const{return shape_[index];}
    inline int ndims() const{return shape_.size();}

    inline int bytes()        const { return bytes_; }
    inline int element_size() const { return data_type_size(dtype_); }
    inline DataHead head()    const { return head_; }

    std::shared_ptr<Tensor> clone() const;
    Tensor& release();
    bool empty() const;

    void set_shape(int ndims, const int* dims);
    void set_shape(const std::vector<int>& dims){ set_shape(dims.size(), dims.data()); }
    
    int device() const{return device_id_;}

    void set_stream(cudaStream_t stream){ stream_ = stream; data_->set_stream(stream); }

    Tensor& to_gpu();
    Tensor& to_cpu();

    template<typename ... _Args>
    int offset(int index, _Args ... index_args) const{
        const int index_array[] = {index, index_args...};
        return offset_array(sizeof...(index_args) + 1, index_array);
    }

    int offset_array(const std::vector<int>& index) const;
    int offset_array(size_t size, const int* index_array) const;

    inline void* cpu() const { return data_->cpu(); }
    inline void* gpu() const { return data_->gpu(); }
    
    template<typename DType> 
    inline const DType* cpu() const { return (DType*)cpu(); }
    template<typename DType> 
    inline DType* cpu() { return (DType*)cpu(); }

    template<typename DType, typename ... _Args> 
    inline DType* cpu(int i, _Args&& ... args) { return cpu<DType>() + offset(i, args...); }


    template<typename DType> 
    inline const DType* gpu() const { return (DType*)gpu(); }
    template<typename DType> 
    inline DType* gpu() { return (DType*)gpu(); }

    template<typename DType, typename ... _Args> 
    inline DType* gpu(int i, _Args&& ... args) { return gpu<DType>() + offset(i, args...); }

    template<typename DType, typename ... _Args> 
    inline DType& at(int i, _Args&& ... args) { return *(cpu<DType>() + offset(i, args...)); }
    
    std::shared_ptr<MixMemory> get_data() const { return data_; }

    Tensor& to_half(); //TODO
    Tensor& to_float(); //TODO

    Tensor& from_cv_img     (int n, const cv::Mat& image);
    Tensor& from_norm_cv_img(int n, const cv::Mat& image, float mean[3], float std[3]);

    bool save_to_file(const std::string& file) const;

    void setup_data();

public:
    std::vector<int> shape_;
    size_t bytes_    = 0;
    DataHead head_   = DataHead::Init;
    DataType dtype_  = DataType::Float;
    int device_id_;
    std::shared_ptr<MixMemory> data_;
    cudaStream_t stream_;
};

std::shared_ptr<Tensor> load_from_file(const std::string& file, int device_id = HOST_ID);

#endif