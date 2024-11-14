# sh ./scripts/DM/train_DM_smmnist.sh


SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint /home/logs_training/AE/smmnist/smmnist_test/snapshots/RegionMM.pth \
    --config /home/config/DM/smmnist.yaml \
    --log_dir /home/logs_training/DM/smmnist \
    --device_ids 0 \
    --postfix test