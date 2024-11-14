# sh ./scripts/AE/train_AE_kth.sh

# Training from scratch
python ./scripts/AE/run.py \
    --config /home/AE/kth.yaml \
    --log_dir /home/AE/kth \
    --device_ids 0 \
    --postfix test

# Resuming training from checkpoint
# --checkpoint ./logs_training/AE/<project_name>/snapshots/RegionMM.pth \
# --set-start True