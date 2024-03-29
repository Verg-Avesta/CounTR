# Instructions to run minimal examples of training and testing with CounTR

Here you can find a zip folder with data and weights used in the following examples and their results:
https://drive.google.com/file/d/134ac1x381PsaC3Y6RFXQWzpmlSzaoc3X/view?usp=sharing.

Here is the W&B report relative to the fine-tuning examples:
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


## Commands to run fine-tuning and testing in different fashions, based on the new and updated parameters

### Fine-tuning with data augmentation

The `--do_aug` parameter is by default `True`.
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir results/finetune_aug --title minimal_finetune_aug --resume weights/FSC147.pth --data_path data/fsc/ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4
```
&nbsp;

### Fine-tuning without data augmentation

With the `--no_do_aug` parameter you can disable data augmentation.
```
python -W ignore FSC_finetune_cross.py --epochs 5 --batch_size 4 --lr 1e-5 --output_dir results/finetune_noaug --title minimal_finetune_noaug --resume weights/FSC147.pth --data_path data/fsc/ --anno_file annotations.json --data_split_file train_test_val.json --class_file images.txt --im_dir images --num_workers 4 --no_do_aug
```
&nbsp;

### Zero-shot demo

With `demo_zero.py` you can test CounTR on your own data in an easy manner without providing any exemplar, just an input file or directory. For each input image, it will print the number of counted objects and the elapsed time.

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
In this way the metrics won't be meaningful but it is possible to try the model on new images without the need of labeling them. That is, like similarly to `demo_zero.py`, but you can provide exemplars in the json annotations file.

&nbsp;


### Plot test results

1. Install libraries needed for the plots, possibly in a virtual environment:
    ```
    pip install plotly
    pip install kaleido==0.1.0.post1
    ```
    (other versions of kaleido may work for you, this is what I tested)

2. Go to the `util` directory and launch your Python interpreter:
    ```bash
    cd util
    python
    ```

3. From within Python:
    ```python
    import misc
    misc.log_test_results("../results")
    misc.plot_test_results("../results")
    ```

The resulting plot, saved in the same directory as the results (in our case, `../results`), shows the performance of CounTR according to the three logged metrics (MAE, RMSE, NAE) along the whole set of experiments. It is particularly useful if your experiments are ordered (e.g., you test your model at different steps along the training, or after some modification you applied).

Note that I removed `test_ext_noGT` from the results used for the plot, since without the ground truth the metrics cannot be meaningful.

&nbsp;


### Make a video from the output of the demo

Steps 1 and 2 are the same as the previous example, then:

```python
import misc
misc.frames2vid("../results/demo_zero_dir", "../results/demo.mp4", "viz*.jpg", 3)
```

It is particularly useful if you are testing CounTR on a sequence of frames.
