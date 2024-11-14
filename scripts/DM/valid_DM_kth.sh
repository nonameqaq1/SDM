# sh ./scripts/DM/valid_DM_kth.sh


########################################
# - VideoFlowDiffusion_multi_w_ref
# - DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada
# -------------------------------------
DM_architecture=VideoFlowDiffusion_multi_w_ref_u22
Unet3D_architecture=DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22


CUDA_VISIBLE_DEVICES=0 \
    python ./scripts/DM/valid.py \
    --num_sample_video  100 \
    --total_pred_frames 40 \
    --num_videos        256 \
    --valid_batch_size  2 \
    --random-seed       1000 \
    --DM_arch           $DM_architecture \
    --Unet3D_arch       $Unet3D_architecture \
    --dataset_path      /home/dataset/kth_h5 \
    --flowae_checkpoint /home/logs_training/AE/kth/RegionMM.pth \
    --config            /home/kth64.yaml \
    --checkpoint        /home/logs_training/DM/kth/kth64_test/snapshots/flowdiff.pth \
    --log_dir           /home/logs_training/val_DM/kth
# --random_time \
