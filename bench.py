import os
import time
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'  # Ampere

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2', '-use_fast_math'])

# GPT2 parameters. Slower if seq_len is too big.
batch_size = 8
n_head = 12
seq_len = 256
head_embd = 64

manual_times, flash_times = [], []
for _ in range(10):
    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    # profiling manual attention
    t0 = time.time()
    manual_result = manual_attn(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    manual_times.append(t1-t0)

    # profiling our minimal flash attention
    t0 = time.time()
    minimal_result = minimal_attn.forward(q, k, v, True)  # Change to False to disable tensor cores (warning: slower)
    torch.cuda.synchronize()
    t1 = time.time()
    flash_times.append(t1-t0)

# Print average elapased time
print('---- profiling results ----')
print(f'manual attention: {sum(manual_times)*1000/len(manual_times):.3f} ms')
print(f'flash attention: {sum(flash_times)*1000/len(flash_times):.3f} ms')

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
