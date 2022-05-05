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
Batch size = 8, accum_iter = 2.


Codes about pre-processing data in FSC147 are in file *FSC147.py*. In *FSC147.py*, only *TransformTrain* and *class resizeTrainImage* are used in fientuning.

|  Task   | model file | train file |
|  ----  | ----  | ----  |
| Pretrain on ImageNet | models_mae.py | main_pretrain.py & engine_pretrain.py |
| Pretrain on FSC147 | models_mae_noct.py | FSC_pretrain.py |
| Finetune on FSC147 | models_mae_up.py | FSC_finetune.py |
