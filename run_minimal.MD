
Finetuning con data augmentation:
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir ./results/finetune_aug --title finetune_aug --resume .\weights\FSC147.pth --data_path .\data\fsc\ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4
```

Finetuning senza data augmentation:
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir ./results/finetune_noaug --title finetune_noaug --resume .\weights\FSC147.pth --data_path .\data\fsc\ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4 --no_do_aug
```

Zero shot demo (directory):
```
python demo_zero.py --input_path data/fsc/images --output_path results/demo_zero_dir
```

Test zero shot:
```
...
```

Test few shot:
```
...
```

Test few shot external:
```
...
```
