# Deployment

### 环境配置

----

1. Linux下[TensorRT安装](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
   - 安装[CUDA11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive)
   - 安装[cudnn8.6.0](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
   - 安装[TensorRT8.5.2](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

2. Linux下[OpenCV安装](https://zhuanlan.zhihu.com/p/392751819)
   - 利用源码构建[OpenCV4.5.4](https://github.com/opencv/opencv/releases)
3. 安装Pytorch（平台不限）
4. 安装ONNX Simplifier

### 部署流程

----

1. Python导出
   - 将Pytorch模型导出ONNX模型
   - 利用ONNX Simplifier简化ONNX模型
2. 模型转化
   - 我们利用TensorRT提供的trtexec将ONNX模型导出为Tensor Engine
3. 模型推理

