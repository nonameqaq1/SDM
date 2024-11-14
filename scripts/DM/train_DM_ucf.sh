# sh ./scripts/DM/train_DM_ucf.sh

SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint /home/logs_training/AE/ucf/RegionMM.pth \
    --config /home/logs_training/AE/ucf/ucf101_64.yaml \
    --log_dir /home/logs_training/DM/ucf \
    --device_ids 0 \
    --postfix test