# flash-attention-minimal
A minimal re-implementation of Flash Attention with CUDA and PyTorch. 
The official [implementation](https://github.com/Dao-AILab/flash-attention) can be quite daunting for a CUDA beginner
(like myself), so this repo tries to be small and educational.

* The entire forward pass is written in ~100 lines in `flash.cu`.
* The variable names follow the notations from the original [paper](https://arxiv.org/abs/2205.14135).

## Usage
Compare the wall-clock time between manual attention and minimal flash attention:
```
python benchmark.py
```

### I don't have a GPU
Check out this [online colab demo](https://colab.research.google.com/gist/tspeterkim/143bc7be7a845656817cf94c5228598e/demo-flash-attention-minimal.ipynb).

## Caveats
* No backward pass!

## References
The paper: https://arxiv.org/abs/2205.14135

Official implementation: https://github.com/Dao-AILab/flash-attention