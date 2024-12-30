# flash-attention-minimal
A minimal re-implementation of Flash Attention with CUDA and PyTorch. 
The official [implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner
(like myself), so this repo tries to be small and educational.

* The entire forward pass is written in ~100 lines in `flash.cu`.
* The variable names follow the notations from the original [paper](https://arxiv.org/abs/2205.14135).

## Usage
### Prerequisite
* PyTorch (with CUDA)
* `Ninja` for loading in C++

### Benchmark
Compare the wall-clock time between manual attention and minimal flash attention:
```
python bench.py
```

Sample output on a [T4](https://aws.amazon.com/ec2/instance-types/g4/):
```
=== profiling manual attention ===
...
Self CPU time total: 52.389ms
Self CUDA time total: 52.545ms

=== profiling minimal flash attention === 
...  
Self CPU time total: 11.452ms
Self CUDA time total: 3.908ms
```
Speed-up achieved! 

### I don't have a GPU
Try out this [online colab demo](https://colab.research.google.com/gist/tspeterkim/143bc7be7a845656817cf94c5228598e/demo-flash-attention-minimal.ipynb).

## Caveats
* No backward pass! To be honest, I found it a lot more complex than the forward pass, which was enough to show the
use of shared memory to avoid large N^2 read/writes.
* In the inner loop, I assign each thread to a row of the output matrix. This differs from the original implementation.
* This thread-per-row simplification makes the matrix multiplications very slow. This is probably why for longer 
sequences and larger block sizes, this gets slower than the manual implementation.
* Q,K,Vs are in float32, unlike the original implementation which uses float16.
* The block size is [fixed](https://github.com/tspeterkim/flash-attention-minimal/blob/9b7ca8ef4e6afdbfeb149a9cd488c8dea9af9ad6/flash.cu#L85) at compile time to 32.

## Todos
- [ ] Add backward pass
- [x] Speed up matmults -> Check out the `tensorcore` branch
- [ ] Dynamically set block size