# sh ./scripts/DM/valid_DM_smmnist.sh


data_path=/home/dataset

# DM_architecture=VideoFlowDiffusion_multi1248
# Unet3D_architecture=DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
DM_architecture=VideoFlowDiffusion_multi_w_ref_u22
Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
SEED=1000
NUM_SAMPLE=100
NUM_BATCH_SIZE=2
########################################

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
# DM_NAME=smmnist64_DM_Batch32_lr2.0e-4_c5p5_STW_adaptor_multi_124_resume
# DM_STEP=flowdiff_0036_S265000
# SEED=1000
# NUM_SAMPLE=100
# NUM_BATCH_SIZE=2
########################################

########################################
# - VideoFlowDiffusion_multi1248
# - DenoiseNet_STWAtt_w_wo_ref_adaptor_cross_multi
# -------------------------------------
# DM_NAME=smmnist64_DM_Batch40_lr2e-4_c10p10_STW_adaptor_multi_1248
# DM_STEP=flowdiff_0040_S195000
# SEED=1000
# NUM_SAMPLE=100
# NUM_BATCH_SIZE=2
########################################

CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/DM/valid.py \
    --num_sample_video  $NUM_SAMPLE \
    --total_pred_frames 10 \
    --num_videos        256 \
    --valid_batch_size  $NUM_BATCH_SIZE \
    --random-seed       $SEED \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      /home/dataset/smmnist_h5 \
    --flowae_checkpoint /home/logs_training/AE/smmnist/smmnist_test/snapshots/RegionMM.pth \
    --config            /home/logs_training/DM/smmnist/smmnist_test/smmnist.yaml \
    --checkpoint        /home/logs_training/DM/smmnist/smmnist_test/snapshots/flowdiff.pth \
    --log_dir           /home/logs_training/val_DM/smmnist