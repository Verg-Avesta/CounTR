# Instructions to run minimal examples of training and testing with CounTR

Here you can find a zip folder with data and weights used in the following examples and their results:
https://drive.google.com/file/d/134ac1x381PsaC3Y6RFXQWzpmlSzaoc3X/view?usp=sharing.

Here is the W&B report relative to the finetuning examples:
https://api.wandb.ai/links/giovanni-ficarra/ilxvm4to.

This should be the resulting structure of the directory, extracting the files from the zip folder into the root of the repository:
```
root
¦   .gitignore
¦   FSC_finetune_CARPK.py
¦   FSC_finetune_cross.py
¦   FSC_pretrain.py
¦   FSC_test_CARPK.py
¦   FSC_test_cross(few-shot).py
¦   FSC_test_cross(zero-shot).py
¦   LICENSE
¦   README.md
¦   demo.py
¦   models_crossvit.py
¦   models_mae_cross.py
¦   models_mae_noct.py
¦   requirements.txt
¦   run.sh
¦   run_minimal.MD
¦   demo_zero.py
¦   
+---img
¦       
+---util
¦           
+---env
+---weights
¦       FSC147.pth
¦       
+---data
¦   +---fsc
¦   ¦   ¦   train_test_val.json
¦   ¦   ¦   annotations.json
¦   ¦   ¦   annotations_ext.json
¦   ¦   ¦   images.txt
¦   ¦   ¦   annotations_ext_noGT.json
¦   ¦   ¦   
¦   ¦   +---images
¦       
+---results
¦   +---demo_zero_dir
¦   +---test_few
¦   +---test_zero
¦   +---test_ext_1
¦   +---test_ext_2
¦   +---test_ext_noGT
```
&nbsp;


## Commands to run finetuning and testing in different fashions, based on the new and updated parameters

### Finetuning with data augmentation

The `--do_aug` parameter is by default `True`.
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir results/finetune_aug --title minimal_finetune_aug --resume weights/FSC147.pth --data_path data/fsc/ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4
```
&nbsp;

### Finetuning senza data augmentation

With the `--no_do_aug` parameter you can disable data augmentation.
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir results/finetune_noaug --title minimal_finetune_noaug --resume weights/FSC147.pth --data_path data/fsc/ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4 --no_do_aug
```
&nbsp;

### Zero-shot demo

With `demo_zero.py` you can test CounTR on your own data in an easy manner without providing any exemplar, just an input file or directory. Forr each input image, it will print the number of counted objects and the elapsed time.

With a single file:
```
python demo_zero.py --input_path data/fsc/images/19.jpg --output_path results/demo_zero_file
```
&nbsp;

With a directory:
```
python demo_zero.py --input_path data/fsc/images --output_path results/demo_zero_dir
```
&nbsp;

### Test zero-shot
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_zero --resume weights/FSC147.pth --box_bound 0
```
&nbsp;

### Test few-shot
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_few --resume weights/FSC147.pth --box_bound 3
```
&nbsp;

### Test few-shot external

Ext 1 - 3 exemplars from a single image:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations_ext.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_ext_1 --resume weights/FSC147.pth --box_bound 3 --external
```
&nbsp;

Ext 2 - 5 exemplars from 3 images:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations_ext.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_ext_2 --resume weights/FSC147.pth --box_bound 5 --external
```
&nbsp;

### Test few shot external noGT

Without providing the ground truth density maps:
```
python 'FSC_test_cross(few-shot).py' --data_path data/fsc --anno_file annotations_ext_noGT.json --data_split_file train_test_val.json --im_dir images --output_dir results/test_ext_noGT --resume weights/FSC147.pth --box_bound 3 --external
```
In this way the metrics won't be meaningful but it is possible to try the model on new images with 
