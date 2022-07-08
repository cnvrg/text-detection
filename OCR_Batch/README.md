# Optical Character Recognition (Batch Predict)
Optical character recognition or optical character reader is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo or from subtitle text superimposed on an image. 

There are several different libraries which can solve the OCR problem, including Google's Tesseract, Amazon's Textract, ABBY Fine Reader and Google's Cloud Vision. Most of them operate from the same basic principle but vary in their parsing of the various fonts and dialects.
### Features
- Upload the images with text and get the text detected in a bounding box as well the parsing of the text in simple language
- User can choose to get specific languages detected and recognized by choosing the arguments

# Input Arguments
- `--images` refers to the name of the path of the directory where images are stored.
- `--lang_list` list of language code you want to recognize, for example 'ch_sim','en'. List of supported language code is here.
- `--decoder` decoder algorithm (default = 'wordbeamsearch') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
- `--beamWidth` beamWidth (default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
- `--contrast_ths` (default = 0.1)-Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to 'adjust_contrast' value. The one with more confident level will be returned as a result.
- `--adjust_contrast` (float, default = 0.5) - target contrast level for low contrast text box
- `--text_threshold` refers to the certainity required for a something to be classified as a letter. The higher this value the clearer characters need to look.
- `--link_threshold` amount of distance allowed between two characters for them to be seen as a single word. Higher the distance, higher probability of different sentences to be classified as a single sentence.
- `--mag_ratio` refers to the image magnification ratio
- `--height_ths` (default = 0.5) refers to the Maximum different in box height. Boxes with very different text size should not be merged.
- `--width_ths` (default = 0.5) - Maximum horizontal distance to merge boxes. ..

# Model Artifacts
- `--output.csv` refers to the name of the file which contains the detected text and the bounding box coordinates, arranged by their file names (and the count of detection)
    |file_name   |text   |x_coord   |y_coord   |width   |height
    |---|---|---|---|---|---|
    |img1.jpg   |Boy was walking   |91   |145   |22  |17   |
    |img2.jpg   |Peter is a lad   |100   |96   |27   |66   |
- `--img1.jpg` refers to the name of the image which has bounding boxes drawn over it

## How to run
```
python3 easy_ocr.py --img_dir /data/imagdata/ --lang_list en --decoder wordbeamsearch --beamWidth 10 --contrast_ths 0.5 --adjust_contrast 0.5 --text_threshold 0.5 --link_threshold 0.5 --mag_ratio 1 --height_ths 0.5 --width_ths 0.5
```

### Reference
https://github.com/JaidedAI/EasyOCR/