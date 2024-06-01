# clears the gpu memory
import gc

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    clear_gpu_memory()
    print("GPU memory cleared.")
else:
    print("CUDA is not available.")


# tells what are the variables is inside gpu memory
import torch
import gc

def get_gpu_memory_usage():
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    return allocated_memory, reserved_memory

def list_gpu_variables():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Tensor on GPU: {obj}, Size: {obj.size()}, Memory: {obj.element_size() * obj.nelement()}")
        except Exception as e:
            pass

if torch.cuda.is_available():
    allocated, reserved = get_gpu_memory_usage()
    print(f"Allocated GPU memory: {allocated} bytes")
    print(f"Reserved GPU memory: {reserved} bytes")
    list_gpu_variables()
else:
    print("CUDA is not available.")
