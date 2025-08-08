# Cuda

## 什么是CUDA？

CUDA (Compute Unified Device Architecture) 是NVIDIA推出的**并行计算平台和编程模型**，使开发者能够利用 NVIDIA GPU 进行高性能计算，或者说CUDA是一个硬件加速平台，提供 GPU 计算能力。
在深度学习项目中，通常使用 Conda 创建虚拟环境，并在其中安装适合的 CUDA 版本，以确保项目的依赖和环境的隔离。
主要特点包括：
1. **GPU计算**：允许开发者使用NVIDIA GPU进行通用计算（GPGPU）
2. **C/C++扩展**：提供对C/C++的扩展，使开发者能直接利用GPU的强大计算能力
3. **生态系统**：包含编译器、库、开发工具和运行时环境

---

## CUDA的安装位置

### 标准安装位置（Linux系统）

```bash
/usr/local/cuda-X.Y          # 具体版本目录
/usr/local/cuda              # 符号链接(指向当前使用的版本)
/usr/local/cuda/bin          # 包含nvcc编译器
/usr/local/cuda/lib64        # 库文件
/usr/local/cuda/include      # 头文件
/usr/local/cuda/samples      # 示例代码
```
### cuda常用的bash命令
- ```nvcc --version```：查看cuda版本  
- ```nvidia-smi```：查看GPU信息，此命令显示 GPU 的使用情况、驱动版本、CUDA 版本等信息  
- ```lspci | grep -i nvidia```：查看GPU型号，此命令列出所有NVIDIA显卡设备  
- 可以在标准安装位置前加```ls -l ```实现查看对应内容的功能：
```bash
ls -l /usr/local/cuda
ls -l /usr/local/cuda/bin
ls -l /usr/local/cuda/lib64
ls -l /usr/local/cuda/include
ls -l /usr/local/cuda/samples
```
例如：
```bash
(base) hy@hy:~$ ls -l /usr/local/cuda
lrwxrwxrwx 1 root root 21 Aug  6 09:00 /usr/local/cuda -> /usr/local/cuda-11.8/
```

### 如何下载不同版本的cuda？
可以从官网下载：  
- cuda11.3：  
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```
- cuda11.6：  
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run
```
- cuda11.8：
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```
