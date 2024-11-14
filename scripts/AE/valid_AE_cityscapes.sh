# sh ./scripts/AE/valid_AE_cityscapes.sh

python valid.py \
    --cond_frames 2 \
    --pred_frames 28 \
    --num_videos 256 \
    --batch_size 256 \
    --input_size 128 \
    --log_dir "/home/cityscapes_test" \
    --data_dir /home/fssd/dataset/cityscapes_h5 \
    --config_path "/home/fssd/cityscapes128.yaml" \
    --restore_from /home/fssd/RegionMM_0128_S150000.pth \
    --data_type "val" \
    --save-video True \
    --random-seed 1000 \
    --gpu "0"