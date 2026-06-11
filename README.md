# Enhancing ROP Screening Reliability with Deep Learning and Uncertainty Estimation

**Beta-EDL**: An evidential deep learning uncertainty-aware model based on self-supervised ConvMAE, enabling reliable AI-assisted referral decisions.

### Key Features
- **Self-supervised Pre-training**: ConvMAE pre-trained on 168,713 unlabeled images
- **Uncertainty Estimation**: EDL-based approach identifies high-uncertainty cases for expert review
- **Human-AI Collaboration**: Uncertain cases referred to specialists, improving screening reliability
- **Multiple Modes**: Normal, MC Dropout, and EDL fine-tuning support

---

## Quick Start

This project provides fine-tuning commands for ConvMAE model with multiple uncertainty estimation modes:

1. **Pretrain** - Self-supervised pre-training using Masked Autoencoder
2. **Normal Mode** - Standard CrossEntropyLoss training + inference
3. **MC Dropout Mode** - Uncertainty estimation via Monte Carlo Dropout + inference
4. **Deep Ensemble Mode** - Uncertainty estimation via checkpoint-based ensemble + inference
5. **EDL Mode** - Uncertainty estimation via Evidential Deep Learning + inference

---

## 1. Self-supervised Pre-training

Pre-train ConvMAE model using Masked Autoencoder (MAE) approach on ImageNet.

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
--batch_size 64 \
--model convmae_convvit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 800 \
--warmup_epochs 20 \
--blr 1e-3 --weight_decay 0.05 \
--data_path /path/to/images/ \
--output_dir ./output_dir/pretrain/
```

---

## 2. Normal Mode

Standard fine-tuning with CrossEntropyLoss.

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
--batch_size 32 \
--model convvit_base_patch16 \
--finetune /path/to/pretrained/model.pth \
--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 \
--data_path /path/to/your/dataset/ \
--output_dir ./output_dir/Normal/ \
--log_dir ./output_dir/Normal/
```

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
--eval --batch_size 128 --nb_classes 2 --seed 42 \
--model convvit_base_patch16 \
--resume /path/to/model/checkpoint.pth \
--data_path /path/to/test/dataset/ \
--eval_save_path ./output_dir/Normal_results.csv
```

---

## 3. MC Dropout Mode

Uncertainty estimation via Monte Carlo Dropout.

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
--batch_size 32 \
--model convvit_base_patch16 \
--finetune /path/to/pretrained/model.pth \
--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 --mc_dropout 0.1 \
--data_path /path/to/your/dataset/ \
--output_dir ./output_dir/MC_Dropout/ \
--log_dir ./output_dir/MC_Dropout/
```

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
--eval --batch_size 128 --nb_classes 2 --seed 42 --mc_dropout 0.1 --mc_dropout_samples 20 \
--model convvit_base_patch16 \
--resume /path/to/model/checkpoint.pth \
--data_path /path/to/test/dataset/ \
--eval_save_path ./output_dir/MC_Dropout_results.csv
```

---

## 4. EDL Mode (Evidential Deep Learning)

Uncertainty estimation via Evidential Deep Learning with Dirichlet distribution.

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
--batch_size 32 \
--model convvit_base_patch16 \
--finetune /path/to/pretrained/model.pth \
--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 --use_edl --layer_decay 1.0 --weight_decay 1e-4 \
--data_path /path/to/your/dataset/ \
--output_dir ./output_dir/EDL/ \
--log_dir ./output_dir/EDL/
```

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
--eval --batch_size 128 --nb_classes 2 --seed 42 --use_edl \
--model convvit_base_patch16 \
--resume /path/to/model/checkpoint.pth \
--data_path /path/to/test/dataset/ \
--eval_save_path ./output_dir/EDL_results.csv
```

---

## 5. Deep Ensemble Mode

Uncertainty estimation via checkpoint-based Deep Ensemble. Selects top-M checkpoints with highest validation AUC scores from training trajectory.

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
--batch_size 32 \
--model convvit_base_patch16 \
--finetune /path/to/pretrained/model.pth \
--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 \
--data_path /path/to/your/dataset/ \
--output_dir ./output_dir/Deep_Ensemble/ \
--log_dir ./output_dir/Deep_Ensemble/
```

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
--eval --batch_size 128 --nb_classes 2 --seed 42 --deep_ensemble --ensemble_models 5 \
--model convvit_base_patch16 \
--resume /path/to/model/checkpoint.pth \
--data_path /path/to/test/dataset/ \
--eval_save_path ./output_dir/Deep_Ensemble_results.csv
```

**Note**: Deep Ensemble uses multiple checkpoints from training. Select checkpoints with highest validation AUC and place them in the checkpoint directory before inference.

---

## Dataset Structure

```
/path/to/dataset/
├── train/
│   ├── class_0/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_1/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class_0/
    └── class_1/
```

---

## Output Format

### Normal
| Column | Description |
|--------|-------------|
| `path` | Image path |
| `labels` | Ground truth label |
| `prob1` | Probability for positive class |

### MC Dropout / EDL
| Column | Description |
|--------|-------------|
| `path` | Image path |
| `labels` | Ground truth label |
| `prob1` | Probability for positive class |
| `uncertainty` | Uncertainty score (higher = more uncertain) |

---

## Environment Setup

### Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+
- scikit-learn, scipy, tensorboard

### Installation
```bash
pip install torch torchvision
pip install scikit-learn scipy tensorboard
```
