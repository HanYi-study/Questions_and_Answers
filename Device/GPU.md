# GPU相关知识

## <mark>GPU详解</mark>
**1.GPU核心思想**：  
**并行**是核心思想，适合 高度并行化 的任务：矩阵乘法、卷积、点云计算、渲染像素。  
- CPU：少核 + 强核 → 几个核心，每个核心很强，适合复杂逻辑、分支判断。
- GPU：多核 + 弱核 → 成百上千个核心，每个核心比较简单，但可以同时算一大堆东西。

**2.GPU硬件架构**：  
以NVIDIA GPU为例：<mark>需要进一步学习</mark>  
```text
GPU 芯片
 ├── SM (Streaming Multiprocessor，多组计算单元)
 │    ├── CUDA cores (标量运算单元)
 │    ├── Tensor Cores (矩阵/AI专用单元)
 │    ├── Warp Scheduler (调度器)
 │    └── 寄存器 / Shared Memory
 ├── L2 Cache
 └── 显存 (GDDR6/HBM)
```
- SM（流式多处理器）：GPU 的基本计算单元，可以看成“小CPU集群”。
- CUDA core：负责最基础的浮点/整数计算。
- Tensor core：AI 深度学习专用单元，优化矩阵乘法（FP16, BF16, TensorFloat32）。
- 显存（VRAM）：存储数据和模型参数，带宽极高，但延迟比寄存器大。

**3.GPU执行模型**：
GPU 使用 SIMT（Single Instruction Multiple Threads） 模型：  
- 一组 32 个线程 组成一个 warp（相当于一个最小执行单位）。
- SM 的调度器会让 warp 同时执行同一条指令（但对不同数据）。
- “数据并行”：同一个公式，算一堆数据点。
- 如果 warp 内的线程执行不同的分支（比如 if/else），会导致 warp divergence（分支发散），降低效率。

**4.GPU内存层次**：
内存层次比较复杂，访问速度差异很大：  
- 寄存器（最快）：每个线程私有，速度最快。
- 共享内存（SM 内部，次快）：同一个 block 的线程可以共享，延迟小。
- L1/L2 Cache：缓存数据，提高访存效率。
- 全局显存（VRAM，最慢）：所有线程共享，但延迟大。

GPU 编程（CUDA、OpenCL）的核心优化点之一，就是 减少全局内存访问，多用共享内存。

**5.GPU的并行编程模型**：  
在CUDA中，程序执行粒度如下：
- Thread（线程） → 最小执行单位
- Warp（32个线程） → 硬件调度的基本单位
- Block（线程块） → 一组线程，运行在一个 SM 上，可以共享共享内存
- Grid（线程网格） → 所有 Block 的集合，运行在整个 GPU 上

例：矩阵乘法  
- 可能会给每个线程分配一个矩阵元素；
- 一个 Block 算一个小子矩阵；
- 所有 Block 合起来算完整矩阵。

---

## <mark>GPU驱动</mark>
**1.什么是GPU驱动**：  
GPU 驱动程序（Graphics Processing Unit Driver） 就是操作系统和显卡硬件之间的“翻译官”。  
GPU 本身是硬件芯片，不能直接理解操作系统的指令。驱动的作用就是：
- 把操作系统或软件的指令（如绘制 3D 图形、矩阵计算）翻译成 GPU 硬件能执行的指令；
- 管理 GPU 的资源（显存、计算核心、线程调度等）；
- 提供 API 接口（如 OpenGL、Vulkan、DirectX、CUDA、ROCm 等）。
- 总结：GPU 驱动让硬件显卡能正常工作，并为开发者提供调用 GPU 的软件接口。

**2.GPU主要组成部分**：  
- 内核态驱动（Kernel Driver）
    - 直接与操作系统内核交互，负责显卡的初始化、内存管理、中断处理。
    - 在 Linux 下通常是 .ko 文件，在 Windows 下是 .sys 文件。
- 用户态驱动（User-space Driver / Runtime Library）
    - 提供给应用程序调用的接口库，比如：CUDA Driver / Runtime（NVIDIA）、 ROCm（AMD）、DirectX 驱动库（Windows）
    - 软件通过这些库间接调用 GPU 资源。
- 编译器/工具链支持
    - 例如 CUDA Toolkit 提供的 nvcc 编译器，可以把 CUDA C++ 代码编译成 GPU 可以理解的二进制。
 
**3.GPU驱动和操作系统、应用的关系**：
```text
[ 应用程序 / 深度学习框架 / 游戏 ]
            ↓ 调用API
 [ CUDA / OpenCL / DirectX / OpenGL ]
            ↓
   [ GPU 用户态驱动库 ]
            ↓
   [ GPU 内核态驱动 ]
            ↓
         [ GPU 硬件 ]
```
- 应用程序（如 PyTorch、TensorFlow、游戏）不会直接和硬件打交道，而是通过驱动暴露的 API。
- 驱动是中间桥梁，保证兼容性和高效运行。

**4.常见GPU驱动类型**：
- **NVIDIA 驱动**：
    - CUDA 驱动（计算为主）
    - 图形驱动（游戏、3D 渲染为主）
    - TensorRT、cuDNN 等是基于驱动的扩展库
- **AMD 驱动**：
    - ROCm（用于深度学习、GPU 计算）
    - Radeon 驱动（游戏/图形）
- **Intel GPU 驱动**：
    - iGPU 驱动（集成显卡）
    - oneAPI（计算类库）
 
**5.驱动与CUDA/ROCm的区别**：
- GPU 驱动：负责最底层的硬件通信，让显卡能用。
- CUDA / ROCm：在驱动之上的编程框架（开发者写代码用的工具和 API）。

**6.GPU驱动与深度学习框架流程示意图**：
```text
[ 你的代码 / 深度学习模型 (PyTorch / TensorFlow) ]
                        │
                        ▼
[ 深度学习框架 Backend (ATen, cuDNN, cuBLAS, TensorRT 等) ]
                        │
                        ▼
[ CUDA Toolkit (nvcc 编译器 / CUDA Runtime / CUDA Driver API) ]
                        │
                        ▼
[ GPU 用户态驱动 (libcuda.so / nvcuda.dll) ]
                        │
                        ▼
[ GPU 内核态驱动 (kernel module, 负责内存、调度、中断) ]
                        │
                        ▼
[ GPU 硬件 (显存 VRAM, CUDA cores, Tensor Cores, RT cores) ]
```
**说明：**
- 你的代码：写的 PyTorch 或 TensorFlow 代码，比如 torch.cuda() 或 model.to("cuda")。
- 框架 Backend：框架调用底层库（如 cuDNN 用于卷积，cuBLAS 用于矩阵运算）。
- CUDA Toolkit：提供 API 和编译器，翻译框架的调用成驱动能理解的指令。
- 用户态驱动：应用层接口，像 libcuda.so（Linux）或 nvcuda.dll（Windows）。
- 内核态驱动：操作系统层面，直接和硬件通信。
- GPU 硬件：最终执行计算，比如矩阵乘法、卷积、光线追踪等。

