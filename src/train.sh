python train.py \
            --train_dir /home/tupm/SSD/CAT_datasets/datasets/train \
            --val_dir /home/tupm/SSD/CAT_datasets/datasets/val \
            --num_workers 8 \
            --num_epochs 30 \
            --batch_size 32 \
            --checkpoint_name '' \
            --image_size 256 \
            --model_name efficientnet-b3
