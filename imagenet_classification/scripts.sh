###########################################################################################
# ResNet-50   (target_rate=0.5, 0.6, 0.7)
## S_net = 1-1-1-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 4-4-2-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 8-4-7-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet50 --finetune_from ckpts/resnet50-19c8e357.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;




###########################################################################################
# ResNet-101   (target_rate=0.4, 0.5, 0.6, 0.7)
## S_net = 1-1-1-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.4 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 4-4-2-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.4 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-4-4-4-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 8-4-7-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.4 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_resnet101 --finetune_from ckpts/resnet101-63fe2227.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 8-8-8-4-4-4-4-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-7-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;




###########################################################################################
# RegNetY-400MF   (target_rate=0.5, 0.6, 0.7)
## S_net = 1-1-1-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 4-4-2-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-1-1-1-1-1-1 \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_400mf --finetune_from ckpts/regnet_y_400mf-c65dace8.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-1-1-1-1-1-1 \s
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


###########################################################################################
# RegNetY-800MF   (target_rate=0.5, 0.6, 0.7)
## S_net = 1-1-1-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 1-1-1-1-1-1-1-1-1-1-1-1-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 4-4-2-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-2-2-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-2-2-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-2-2-2-2-2-2-2-2-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;


## S_net = 4-4-7-1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.5 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-7-7-7-7-7-7-7-7-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.6 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-7-7-7-7-7-7-7-7-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train/main.py \
--train_url YOUR_SAVE_PATH \
--data_url YOUR_DATA_PATH --dataset imagenet --workers 32 --config configs/finetune_100eps_512bs_lr0x04.py \
--arch dyn_regnet_y_800mf --finetune_from ckpts/regnet_y_800mf-1b27b58c.pth --lr_mult 0.1 --T_kd 4.0 --alpha_kd 0.5 \
--target_begin_epoch 0 --target_rate 0.7 --lambda_act 10.0 --temp_scheduler exp --t0 5.0 --t_last 0.1 \
--mask_channel_group 1-1-1-1-1-1-1-1-1-1-1-1-1-1 \
--mask_spatial_granularity 4-4-4-4-7-7-7-7-7-7-7-7-1-1  \
--dist_url tcp://127.0.0.1:10001 --print_freq 100 --round 1;