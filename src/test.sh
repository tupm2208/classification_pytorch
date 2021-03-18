python test.py \
            --test_dir /home/tupm/SSD/CAT_datasets/workspace/smartcare/topics_v3 \
            --num_workers 8 \
            --batch_size 32 \
            --checkpoint_path ../checkpoints/checkpoint_6.pth.tar \
            --test_csv ../datasets/test_fix_2.csv \
            --model_name efficientnet-b3