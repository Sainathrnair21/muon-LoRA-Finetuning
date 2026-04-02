# Muon Optimizer for LoRA Fine-Tuning of Large Language Models

This repository contains the experimental code for my Penn State Schreyer Honors College thesis: **Modern Optimizers for Deep Learning: The Muon Optimizer for Low-Rank Fine-Tuning of Large Language Models.**

These experiments investigate whether LoRA provides implicit robustness to the optimizer mismatch between Muon and AdamW-pretrained models. While the Moonlight paper (2025) uses a carefully tuned learning rate to achieve results on par with AdamW, we show that combining Muon with LoRA achieves stable and competitive training even at suboptimal learning rates. We also analyze the singular values of the LoRA adapter matrices trained by each optimizer to confirm Muon's uniform spectral growth, consistent with Kang et al. (2026).

## Setup

All experiments were run on Lambda Labs GH200 GPU instances (80GB VRAM). A high-memory GPU is strongly recommended as loading 3B+ parameter models in bfloat16 with gradient tracking requires significant VRAM. Running on a GPU with less than 40GB VRAM will likely cause out-of-memory errors.

## Dependencies

Run the following in your notebook or terminal before starting:
```bash
pip install -q transformers peft datasets accelerate trl nbformat
pip install --upgrade Pillow
```

## Muon Implementation

The Muon optimizer is adapted from [Keller Jordan's original implementation](https://github.com/KellerJordan/Muon), extended with the RMS scaling and weight decay improvements introduced by the [Moonlight paper](https://arxiv.org/abs/2502.16982) (Liu et al., 2025).

## Experiments

Each experiment is contained in a separate Jupyter notebook.

| Notebook | Description |
|----------|-------------|
| `experiment_1.ipynb` | Compares Muon+LoRA and AdamW+LoRA on Phi-4-mini and Qwen2.5-3B at fixed learning rates (0.002 and 0.01) at LoRA rank 16. Includes SVD analysis of adapter matrices. |
| `experiment_2.ipynb` | Extends the comparison across LoRA ranks 4, 8, 16, 32, and 64 at fixed LR 0.002 on Phi-4-mini. |
| `experiment_2b_hpsearch.ipynb` | Repeats the rank ablation with per-rank per-optimizer learning rate grid search on both models. |
| `experiment_3.ipynb` | Tests Muon and AdamW under full fine-tuning without LoRA at multiple learning rates, examining the optimizer mismatch problem. |

## Models and Dataset

- **Models:** `microsoft/Phi-4-mini-instruct`, `Qwen/Qwen2.5-3B-Instruct`
- **Dataset:** `zwhe99/commonsense_170k` (via Hugging Face)
- All runs use 3 epochs, effective batch size 64, and sequence length 512.

## Notes

- This implementation uses plain momentum rather than Nesterov momentum as in Jordan et al.'s original.
- Distributed training is not supported. All experiments run on a single GH200 instance.

## References

- Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks. https://github.com/KellerJordan/Muon
- Liu et al. (2025). Muon is Scalable for LLM Training. arXiv:2502.16982
- Kang et al. (2026). Uniform Spectral Growth and Convergence of Muon in LoRA-Style Matrix Factorization. arXiv:2602.06385
