python -m torch.distributed.run --nproc_per_node=4 main.py \
--input_size 64 --epochs 120 \
--batch_size 64 --lr 4e-3 --update_freq 4 \
--data_set TINY_IMNET \
--data_path /mnt/nvme2/xuhaiyang/data/tiny-imagenet-200/ \
--model convnext_small \
--drop_path 0.1 --normalization true \
--weight_decay 0.05 --residual true \
--output_dir outputs/m-s+dp-01+n-T+wd-005+r-T