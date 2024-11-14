# sh ./scripts/DM/train_DM_kth.sh

python ./scripts/DM/run.py \
    --random-seed 1234 \
    --flowae_checkpoint /home/logs_training/AE/kth/RegionMM.pth \
    --config /home/kth64.yaml \
    --log_dir /home/logs_training/DM/kth \
    --device_ids 0 \
    --postfix test