# Roation + Pruning

## Setup

Below is an example command for pruning LLaMA2-7B with Rotation and Sparsegpt, to achieve unstructured 50% sparsity.
Step 1:
In the QuaRot Repo download the fast_hadamard_transform [QuaRot](https://github.com/spcl/QuaRot).
```
python setup.py install
```
Step 2:
Back to original repo
```sh
python main.py     --model meta-llama/Llama-2-7b-hf     --prune_method sparsegpt     --sparsity_ratio 0.5     --sparsity_type unstructured --rotate --reorder
```
We provide a quick overview of the arguments:  
- `--rotate`: Adding roation to Pruning
- `--reorder`: Adding permutation to Pruning

Llama 2 7b model path:
```
ln -s /data/datasets/llama2 ./meta-llama
```
