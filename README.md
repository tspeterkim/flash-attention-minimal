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

```bash
python bench.py
```

Sample output on a [T4](https://aws.amazon.com/ec2/instance-types/g4/) for the forward pass (Br = Bc = 32):

```
=== profiling manual attention (forward pass) ===
...
Self CPU time total: 52.389ms
Self CUDA time total: 52.545ms

=== profiling minimal flash attention (forward pass) === 
...
Self CPU time total: 11.452ms
Self CUDA time total: 3.908ms
```

That's a 13x speedup!

Sample output on an RTX 3060 for the backward pass (Br = Bc = 16):

```
=== profiling manual attention (backward pass) ===
...
Self CPU time total: 11.139ms
Self CUDA time total: 1.721ms

=== profiling minimal flash attention (backward pass) === 
...
Self CPU time total: 31.466ms
Self CUDA time total: 629.000us
```

That's a 2x speedup! Note though that we've only tested this on an RTX 3060 which has a smaller SRAM than the T4
(hence the reduction of block size from 32 to 16). The speedup might be different on a T4.

### I don't have a GPU

Try out this [online colab demo](https://colab.research.google.com/gist/tspeterkim/143bc7be7a845656817cf94c5228598e/demo-flash-attention-minimal.ipynb).

## Caveats

* In the inner loop, I assign each thread to a row of the output matrix. This differs from the original implementation.
* This thread-per-row simplification makes the matrix multiplications very slow. This is probably why for longer
sequences and larger block sizes, this gets slower than the manual implementation.
* Q,K,Vs are in float32, unlike the original implementation which uses float16.
* The block size is [fixed](https://github.com/tspeterkim/flash-attention-minimal/blob/9b7ca8ef4e6afdbfeb149a9cd488c8dea9af9ad6/flash.cu#L85) at compile time to 32.

## Todos

* [ ] Speed up matmults
* [ ] Dynamically set block size

## Contributors

* [Peter Kim](https://github.com/tspeterkim), Lead Contributor
* [Franz Cesista](https://github.com/leloykun), Implemented the backward pass
