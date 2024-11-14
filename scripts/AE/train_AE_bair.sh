# sh ./scripts/AE/train_AE_bair.sh

python ./scripts/AE/run.py \
    --config /home/bair.yaml \
    --log_dir /home/AE/bair \
    --device_ids 0 \
    --postfix test
