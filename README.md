# Roation + Pruning

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

Below is an example command for pruning LLaMA2-7B with Rotation and Sparsegpt, to achieve unstructured 50% sparsity.
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
  
For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss papers/code with us through issues/emails!

mingjies at cs.cmu.edu  
liuzhuangthu at gmail.com 
