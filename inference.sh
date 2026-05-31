
## Normal
#checkpoint='best_checkpoint_0.8305_0'
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
#--eval --batch_size 128 --nb_classes 2 --seed 42 \
#--model convvit_base_patch16 \
#--resume /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/Normal/${checkpoint}.pth \
#--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset_jsiec/random_all/test/ \
#--eval_save_path ./output_dir/Normal_test_${checkpoint}.csv


# MC_Dropout
checkpoint='best_checkpoint_0.9671_5'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
--eval --batch_size 128 --nb_classes 2 --seed 42 --mc_dropout 0.1 --mc_dropout_samples 20 \
--model convvit_base_patch16 \
--resume /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/MC_Dropout/${checkpoint}.pth \
--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset_jsiec/random_all/ \
--eval_save_path ./output_dir/MC_Dropout_val_${checkpoint}.csv


## DEL
#checkpoint='best_checkpoint_0.9597_5'
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
#--eval --batch_size 128 --nb_classes 2 --seed 42 --use_edl \
#--model convvit_base_patch16 \
#--resume /media/ubuntu/hdd/ai_ljw/ROP/ROP_result/for_github_test/fineture/EDL_all/${checkpoint}.pth \
#--data_path /home/ubuntu/ai_ljw/ROP/rw_rop/dataset_jsiec/random_all/ \
#--eval_save_path ./output_dir/EDL_val_${checkpoint}.csv