Use this blueprint with your custom data to train a model that detects text elements in images. This blueprint trains and deploys a fine-tuned model to use for inference with API calls on the endpoint the blueprint also generates.

This blueprint trains a text-detection algorithm on custom fonts. If the default model performs poorly on a set of images due to the font or background, the algorithm can be retrained on your custom images dataset. To fine-tune the model for specific fonts, provide the training dataset containing images of the desired text font. 
The input dataset requires only about 1500 images, which are essentially cutouts of the words written in the desired font. The dataset consists of images and a single, two-column `labels.csv` file, with the first column containing the image filename and the second column containing the corresponding word. Each image in the training dataset must have an associated entry in the `labels.csv` file.

Complete the following steps to train the text-detector model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `bucketname` − Value: enter the data bucket name
     * Key: `prefix` − Value: provide the main path to the images folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     * Key: `language` − Value: enter the language code on which to train the model
     * Key: `data` − Value: provide the path to the images folder including the S3 prefix in the following format: `/input/s3_connector/<prefix>/ocr_data`
     NOTE: You can use the prebuilt example data paths provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained text-detector model and deploying it as a new API endpoint.
5. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   * Use the Try it Live section with any text-containing image to check the model.
   * Use the bottom integration panel to integrate your API with your code by copying in the code snippet.

A custom model and API endpoint, which can detect an image's text elements, have now been trained and deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/text-detection).
