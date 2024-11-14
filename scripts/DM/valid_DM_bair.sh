# sh ./scripts/DM/valid_DM_bair.sh


#
#data_path=/home/ubuntu/zzc/data/video_prediction/dataset_h5
#
#AE_NAME=bair64_scale0.50
#AE_STEP=RegionMM
#
#########################################
## - VideoFlowDiffusion_multi_w_ref
## - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
## -------------------------------------
## pred4 - repeat1  - 256bs - 30966MB
## pred4 - repeat2  - 128bs - 25846MB
## pred4 - repeat4  -  64bs - 26376MB
## -------------------------------------
#DM_architecture=VideoFlowDiffusion_multi_w_ref
#Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
#DM_NAME=bair64_DM_Batch64_lr2e-4_c2p4_STW_adaptor_scale0.50_multi_traj
#DM_STEP=flowdiff_best_73000_315.362
#SEED=1000
#NUM_SAMPLE=100
#NUM_BATCH_SIZE=128
#########################################

########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# -------------------------------------
# pred5 - repeat1  - 256bs - 32474MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_u12
# DM_NAME=bair64_DM_Batch64_lr2.e-4_c2p5_STW_adaptor_multi_traj_resume
# DM_STEP=flowdiff_0064_S190000
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=128
########################################

########################################
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# pred7 - repeat1  - 256bs - 37520MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=bair64_DM_Batch66_lr2.e-4_c2p7_STW_adaptor_multi_traj_ada
# DM_STEP=flowdiff_0066_S095000
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=128
########################################

########################################
#  - VideoFlowDiffusion_multi_w_ref
#  - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
# pred10 - repeat1  - 256bs - 42684MB
# -------------------------------------
# DM_architecture=VideoFlowDiffusion_multi_w_ref
# Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# DM_NAME=bair64_DM_Batch64_lr2e-4_c2p10_STW_adaptor_scale0.50_multi_traj_ada
# DM_STEP=flowdiff_best_239.058
# SEED=1000
# NUM_SAMPLE=1
# NUM_BATCH_SIZE=64
########################################

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/DM/valid.py \
    --num_sample_video  100 \
    --total_pred_frames 28 \
    --num_videos        256 \
    --valid_batch_size  4 \
    --random-seed       1000 \
    --DM_arch           "VideoFlowDiffusion_multi_w_ref_u22" \
    --Unet3D_arch       'DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22' \
    --dataset_path      '/home/dataset/BAIR_h5/' \
    --flowae_checkpoint /home/RegionMM_bair.pth\
    --config            /home/logs_training/DM/BAIR/bair_test/bair.yaml \
    --checkpoint        /home/logs_training/DM/BAIR/bair_test/snapshots/flowdiff.pth \
    --log_dir           ./logs_validation/diffusion/bair/

