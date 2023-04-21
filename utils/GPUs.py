import os
import torch
import warnings
import gpustat


def init_gpu(require_num: int = 1):
    # this must be called before cuda initializing
    try:
        gpu_list = gpustat.GPUStatCollection.new_query()
    except Exception:
        warnings.warn("GPU initialization failed.")
        return None
    print(gpu_list)
    gpu_available = []
    for idx, gpu in enumerate(gpu_list):
        if len(gpu.processes) == 0:      # no process working in the gpu
            gpu_available.append(idx)
    if len(gpu_available) >= require_num:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_available[:require_num]))
        return [gpu_list[idx] for idx in gpu_available[:require_num]]
    else:
        warnings.warn("Not enough GPU available now")
        return None
