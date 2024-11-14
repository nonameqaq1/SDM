# sh ./scripts/AE/train_AE_cityscapes.sh

# Training from scratch
python run.py \
    --config /home/cityscapes128.yaml \
    --log_dir /home/AE/cityscapes \
    --device_ids 0 \
    --postfix test

# Resuming training from checkpoint
# --checkpoint ./logs_training/AE/<project_name>/snapshots/RegionMM.pth \
# --set-start True