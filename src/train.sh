python train.py \
            --train_dir /home/tupm/SSD/CAT_datasets/workspace/smartcare/topics_v3 \
            --val_dir /home/tupm/SSD/CAT_datasets/workspace/smartcare/topics_v3 \
            --num_workers 8 \
            --num_epochs 30 \
            --batch_size 32 \
            --checkpoint_name 'checkpoint_0.pth.tar' \
            --image_size 256 \
            --model_name efficientnet-b3 \
            --train_csv ../datasets/train_fix_2.csv \
            --val_csv ../datasets/val_fix_1.csv 

