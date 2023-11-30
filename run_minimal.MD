
Finetuning con data augmentation:
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir ./results/finetune_aug --title minimal_finetune_aug --resume .\weights\FSC147.pth --data_path .\data\fsc\ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4
```

Finetuning senza data augmentation:
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir ./results/finetune_noaug --title minimal_finetune_noaug --resume .\weights\FSC147.pth --data_path .\data\fsc\ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4 --no_do_aug
```

Zero shot demo (directory):
```
python demo_zero.py --input_path data/fsc/images --output_path results/demo_zero_dir
```

Test zero shot:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_zero --resume weights/FSC147.pth --box_bound 0
```

Test few shot:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_few --resume weights/FSC147.pth --box_bound 3
```

Test few shot external:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations_ext.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_ext_2 --resume weights/FSC147.pth --box_bound 5 --external
```

W&B report:
https://api.wandb.ai/links/giovanni-ficarra/ilxvm4to
