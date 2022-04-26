# MAE_Counting

Run the code by

```
CUDA_VISIBLE_DEVICES=5 nohup python -u FSC_train.py \
    --epochs 50000 \
    --blr 5e-7 --weight_decay 0.05  >>./output_s1_dir/FSC_train.log 2>&1 &
```

Now is doing overfitting experiments without normalization and with encoder fixed.
The density map is scaled up by 100x.
2 batches, 16 images.
