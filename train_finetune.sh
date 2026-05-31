#!/bin/bash

#sudo nvidia-smi -pl 250

## pretrain
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
#--batch_size 64 \
#--model convmae_convvit_base_patch16 \
#--norm_pix_loss \
#--mask_ratio 0.75 \
#--epochs 800 \
#--warmup_epochs 20 \
#--blr 1e-3 --weight_decay 0.05 \
#--data_path /media/ubuntu/hdd/ai_ljw/ROP/preprocess/ \
#--output_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/pretrain/

## Normal
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
#--batch_size 32 \
#--model convvit_base_patch16 \
#--finetune /home/ubuntu/ai_ljw/ROP/convmae/convmae_rop.pth \
#--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
#--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 \
#--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset_jsiec/random_all/ \
#--output_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/Normal/ \
#--log_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/Normal/


# MC-Dropout
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
--batch_size 32 \
--model convvit_base_patch16 \
--finetune /home/ubuntu/ai_ljw/ROP/convmae/convmae_rop.pth \
--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 --mc_dropout 0.1 \
--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset_jsiec/random_all/ \
--output_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/MC_Dropout/ \
--log_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/MC_Dropout/



## EDL
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
#--batch_size 32 \
#--model convvit_base_patch16 \
#--finetune /home/ubuntu/ai_ljw/ROP/convmae/convmae_rop.pth \
#--epochs 50 --blr 1e-4 --reprob 0.0 --aa '' \
#--nb_classes 2 --seed 42 --warmup_epochs 2 --clip_grad 0.5 --use_edl --layer_decay 1.0 --weight_decay 1e-4 \
#--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset/random_all/ \
#--output_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/EDL_all/ \
#--log_dir /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/EDL_all/