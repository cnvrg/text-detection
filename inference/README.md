# Text Detection Inference
Text Detection is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo or from subtitle text superimposed on an image. 

This library deploys the Text Detection model which you can immediately use to run inferences on your images. By default `English` language model is loaded. In case your images have text from different languages, you can choose to set the the environment variables in cnvrg and provide the list of all languages in your images.
You can set the environment variables as follows:
1. Open your project.
2. Go to `settings` tab for the project.
3. Choose `Environment` option to open environment settings.
4. Enter the `key` and `value` for the `Environment variable` you would like to set.

For example for in case your images have English and Chinese text then enter the following values in `key` and `value` fields in `Environment variable` option:
`key`: lang_list
`value`: en,ch_sim

**Note**: Currently we only support inference on english model for pretrained models. In case you have trained the model on your own data, in that case you can refer the list of languages supported here. [here](https://www.jaided.ai/easyocr/).(Scroll down the page to see the list) 
Similarly you can choose to change the default for certain arguments to the Text Detection model by clicking `+` and adding more environment variables. Different parameters are explained below:

- `lang_list` list of language code you want to recognize, for example 'ch_sim','en'. Currently only english supported for pretrained models. (Models that were not trained by you.)
- `decoder` decoder algorithm (default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
- `beamWidth` beamWidth (default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
- `contrast_ths` (default = 0.1)-Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to 'adjust_contrast' value. The one with more confident level will be returned as a result.
- `adjust_contrast` (float, default = 0.5) - target contrast level for low contrast text box
- `text_threshold` refers to the certainity required for a something to be classified as a letter. The higher this value the clearer characters need to look.
- `link_threshold` amount of distance allowed between two characters for them to be seen as a single word. Higher the distance, higher probability of different sentences to be classified as a single sentence.
- `mag_ratio` refers to the image magnification ratio
- `height_ths` (default = 0.5) refers to the Maximum different in box height. Boxes with very different text size should not be merged.
- `width_ths` (default = 0.5) - Maximum horizontal distance to merge boxes. ..

An example json response for two images from the endpoint looks like below:
```
{
    1: [
        {
                "text": "apple",
                "bounding_box": [80, 100, 20, 25],
                "confidence": 0.97,
            },
            {
                "text": "banana",
                "bounding_box": [55, 106, 40, 35],
                "confidence": 0.67,
            }
    ],
    2: [
        {
                "text": "Eating",
                "bounding_box": [80, 90, 30, 25],
                "confidence": 0.87,
            },
            {
                "text": "healthy",
                "bounding_box": [50, 10, 41, 25],
                "confidence": 0.74
            }
    ]
}
```

### Reference
https://github.com/JaidedAI/EasyOCR/









