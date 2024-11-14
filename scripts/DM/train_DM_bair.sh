# sh ./scripts/DM/train_DM_bair.sh

SEED=1234

python ./scripts/DM/run.py \
    --random-seed $SEED \
    --flowae_checkpoint /home/logs_training/AE/bair/bair_test/snapshots/RegionMM.pth \
    --config /home/sdm/config/DM/bair.yaml \
    --log_dir /home/logs_training/DM/BAIR \
    --device_ids 0 \
    --postfix test