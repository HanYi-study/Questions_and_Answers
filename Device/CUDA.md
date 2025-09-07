# 从学习角度看CUDA

## 1.CUDA是什么？
**CUDA（Compute Unified Device Architecture） 是 NVIDIA 提供的一套 GPU 编程平台**，CUDA 是 NVIDIA 的 GPU 编程生态，让你写代码用显卡来算，而不仅仅是显示画面。  
它包括：
- CUDA 驱动（Driver API）：最底层接口，直接和 GPU 驱动交互。
- CUDA Runtime（运行时库）：封装了 Driver API，更方便开发者调用。
- nvcc 编译器：可以把 CUDA C/C++ 代码编译成 GPU 能运行的二进制。
- 工具和库：比如 cuBLAS（矩阵库）、cuDNN（深度学习加速库）。

## 2.为什么会有不同版本CUDA？
- 硬件和驱动更新：新显卡需要支持新的 CUDA 版本；老 CUDA 版本可能不认识新硬件。
- 软件依赖：比如 PyTorch 2.0 编译时用 CUDA 11.8，那么你本地也得有 CUDA 11.8 才能运行（否则找不到函数）。
- 兼容性问题：有些库只支持特定的 CUDA 版本，比如 TensorFlow 2.4 要 CUDA 11.0，而 PyTorch 1.10 可能要 CUDA 11.3。

## 3.CUDA存在于什么层次？
需要区分**全局安装**和**项目使用**：  
- CUDA 驱动（Driver）：系统级别的，只需要一个，所有 CUDA 程序共用，（比如 /usr/lib/nvidia-driver-XXX）。
- CUDA Toolkit（nvcc、runtime、库）：可以有多个版本同时存在（比如 /usr/local/cuda-10.2，/usr/local/cuda-11.8）。
- 项目层面：项目并不会“自带 CUDA”，而是依赖某个系统里安装的 CUDA 版本，或者在虚拟环境里通过预编译的二进制（比如 pip install torch==2.0.1+cu118）。

总结：
- 驱动是全局唯一的，驱动（Driver）就像显卡的“操作系统”，统一安装一次，全局使用。
- CUDA toolkit 就像开发工具链，可以装多个版本，项目里选择合适的版本来用。
- 项目通过虚拟环境来绑定某个 CUDA 版本。
- Python 项目就像应用程序，需要调用不同版本的 CUDA 库来运行。

## 4.为什么不同项目需要不同 CUDA 版本？
- PyTorch / TensorFlow 在编译时会绑定某个 CUDA 版本（比如 +cu118 就是 CUDA 11.8）。
- 如果你的系统 CUDA 不匹配，程序运行时可能报错（找不到符号，或者算子不兼容）。
- 所以不同项目选择的框架版本不一样 → 要的 CUDA 版本也就不同。

## 5. 如何让不同项目指向不同 CUDA 版本？
### 方法一：Conda / pip 安装带 CUDA 的预编译包（推荐）
例如安装pytorch的时候：  
```bash
# CUDA 11.8 版本
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 12.1 版本
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
这种方式 不依赖系统 CUDA，每个虚拟环境里自带所需 CUDA 库。

### 方法二：系统多版本 CUDA Toolkit 并切换
- 可能你的服务器上 /usr/local/cuda-10.2，/usr/local/cuda-11.8 都存在。
- 可以用环境变量切换：
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
- 在不同项目的虚拟环境 activate 脚本<mark>（什么是activate脚本？在哪里找？）</mark>里写好，就能自动切换。

## 6.GPU 驱动 & CUDA & Python 项目关系图
```text
┌─────────────────────────────┐
│       你的 Python 项目       │
│  (PyTorch / TensorFlow 等)   │
│   └── 运行在 Conda/venv 中   │
└─────────────▲───────────────┘
              │ 调用
┌─────────────┴───────────────┐
│   CUDA Toolkit (用户态库)    │
│  - nvcc 编译器               │
│  - CUDA Runtime (libcudart)  │
│  - CUDA Libraries (cuBLAS,   │
│    cuDNN, NCCL 等)           │
│  → 可以多版本共存            │
│  → 项目通过 PATH/LD_LIBRARY  │
│    环境变量选择使用哪个版本   │
└─────────────▲───────────────┘
              │ 调用
┌─────────────┴───────────────┐
│   CUDA Driver API (系统级)   │
│  - nvidia-driver (内核态)    │
│  - libcuda.so / nvcuda.dll   │
│  → 全局唯一，和 GPU 硬件绑定 │
│  → 新驱动通常兼容旧 CUDA     │
└─────────────▲───────────────┘
              │
┌─────────────┴───────────────┐
│        GPU 硬件 (NVIDIA)     │
│  - CUDA cores / Tensor Cores │
│  - 显存 VRAM                 │
└─────────────────────────────┘
```
