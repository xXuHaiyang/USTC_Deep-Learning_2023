python -m torch.distributed.run --nproc_per_node=4 main.py \
--input_size 192 --epochs 240 \
--batch_size 32 --lr 4e-3 --update_freq 4 \
--data_set TINY_IMNET \
--data_path /mnt/nvme2/xuhaiyang/data/tiny-imagenet-200/ \
--model convnext_base \
--drop_path 0.2 --normalization true \
--weight_decay 0.05 --residual true \
--use_amp true \
--output_dir outputs/m-b+dp-02+n-T+wd-005+r-T+e-240+i-192