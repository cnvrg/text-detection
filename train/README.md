# Text Detection Training

This library trains the Text Detection algorithm on custom fonts. In case the default model is performing poorly on your given set of images due to a specific font or background, you can choose to retrain the Text Detection algorithm on the dataset similar to your images.

All you need is approximately 1500 images, which are essentially cutout of the words written in the font of your choice. For example below

The datasets consists of images and a single `labels.csv` file. This file contains the text in the training images. The structure of this file is below.
| filename | words |
| :---:   | :---:   |
| img0.jpg | tel |
| img1.jpg | apple |
| img2.jpg | reapply |


For each image in the training dataset there must be a corresponding entry in the `labels.csv` file.

### Input
- `--language` Specify the language code to select the language model you want to retrain.
- `--iterations` specify the number of iterations for which you want to retrain the model. Default value is 300000. Higher number results in higher accuracy but too high a number can lead to overfitting and your model will fail to generalize. The number of iterations set depend on the training and val error. As the training progress, ideally both the losses should come down and after a certain point, val error starts increasing and that is when we should stop the training. Usually data that is of larger size requires more iterations.
- `--data` specify the folder location containing all images and the labels file as specified above.
  
### Output
The final output contains a `custom_model.pth`, `custom_model.py` and `custom_model.yaml` files. These three files are used to load the new model and run inferences.

### How to run

```
cnvrg run  --datasets='[{ id:{dataset name},commit:{commit ID } }]' --machine={comput size} --image={docker image} --sync_before=false python3 train.py --language {language of the model} --iterations {number of iterations} --data {dataset name}
```
Example run

```
cnvrg run  --datasets='[{id:"ocrdata",commit:"2659fc519890c924f82b4475ddd71b058178d02b"}]' --machine="default.gpu-small" --image=cnvrg_gpu:nvidia-tf-19.10 --sync_before=false python3 train.py --language en --iterations 100000 --data ocrdata

```

### Reference
https://github.com/JaidedAI/EasyOCR/