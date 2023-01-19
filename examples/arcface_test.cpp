#include"src/application/arcface/arcface.h"
#include"src/application/cuda_tools.h"

int main(int argc, const char *argv[]){
    int batch_size = 1;
    Arcface instance(batch_size);
    instance.create_infer("../trt_model/arcface.trt",0);
    
    std::vector<img_bin> imgs_bin;
    for(int i=0;i<batch_size;i++){
        std::string img_path;
        std::cout << "Please input image path: " << std::endl;
	    std::cin >> img_path;
        imgs_bin.push_back(CUDA_TOOLS::load_file(img_path));
    }

    instance.set_input(imgs_bin, 0);
    instance.inference();

    std::shared_ptr<Tensor> output = instance.get_output(0); //gpu

    cudaStream_t stream = nullptr;
	checkCudaRuntime(cudaStreamCreate(&stream));
    output->set_stream(stream);

    output->to_cpu();

    for(int i=0;i<batch_size;i++){
        float* feature = output->cpu<float>(i);

        int feature_dim = output->shape(1);
        for(int i=0;i<feature_dim;i++)
            printf("%f\t", feature[i]);
        printf("\n");
    }

    checkCudaRuntime(cudaStreamDestroy(stream));
    
    return 0;
}