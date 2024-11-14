# sh ./scripts/DM/train_DM_cityscapes.sh

SEED=1234

python ./scripts/DM/run.py \
    --random-seed 1234 \
    --flowae_checkpoint /home/logs_training/AE/cityscapes/RegionMM.pth \
    --config /home/cityscapes128.yaml \
    --log_dir /home/logs_training/DM/cityscapes \
    --device_ids 0 \
    --postfix test