# pytt_loader.py
import os
import sys
import importlib.util
import ctypes

def load_pytt():
    """自动设置环境并加载 pytt 模块"""
    # 1. 获取 CUDA_PATH 环境变量
    cuda_path = os.getenv("CUDA_PATH")
    if not cuda_path:
        raise EnvironmentError("CUDA_PATH 环境变量未设置，且未找到默认 CUDA 安装")
    cuda_bin = os.path.join(cuda_path, "bin")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_bin)
    else:
        # 旧版 Python 添加到 PATH
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
    os.add_dll_directory(r"D:\tools\opencv\build\bin")
    os.add_dll_directory(r"E:\C_project\TinyTorch-official\TinyTorch-main\thirdparty\OpenBLAS\win64-64\bin")  # OpenBLAS
    os.add_dll_directory(r"E:\C_project\TinyTorch-official\TinyTorch-main\cmake-build-debug-visual-studio\pybind\src")  # 模块所在目录
    # 3. 预加载关键 DLL
    required_dlls = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cufft64_12.dll",
        "curand64_12.dll"
    ]
    for dll in required_dlls:
        dll_path = os.path.join(cuda_bin, dll)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
            except OSError as e:
                sys.stderr.write(f"警告: 无法预加载 {dll}: {e}\n")

    # 4. 加载 pytt 模块
    module_path = r"E:\C_project\TinyTorch-official\TinyTorch-main\cmake-build-debug-visual-studio\pybind\src\pytt.pyd"

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"找不到 pytt 模块: {module_path}")

    spec = importlib.util.spec_from_file_location("pytt", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

pytt = load_pytt()