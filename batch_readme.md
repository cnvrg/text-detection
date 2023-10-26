Use this blueprint to detect text elements in a batch of images. To use this blueprint, provide one `img_dir` folder in the S3 Connector containing the images on which to detect the text.

The text-detection algorithm can be fine-tuned for custom fonts in the event the default text-detector model does not recognize a user’s desired font. For more information, see this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/text-detection-training).

Complete the following steps to run the text-detector model in batch mode:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` − Value: provide the data bucket name
     - Key: `prefix` − Value: provide the main path to the images folders
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Batch-Inference** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `img_dir` − Value: provide the path to the images directory including the S3 prefix in the following format: ` /input/s3_connector/text_detection_batch_data` 
     - Key: `--lang_list` − Value: list the language code to recognize, for example `en`
     - Key: `--decoder` − Value: provide the decoder algorithm, for example `wordbeamsearch` (default)
     NOTE: You can use the prebuilt example data paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a text-detector model that detects text in a batch of images and downloads a CSV file with the information on the text and bounding boxes.
5. Track the blueprint’s real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Select **Batch Inference > Experiments > Artifacts** and locate the bounding box images and output CSV file.
7. Click the **output.csv** File Name, click the right Menu icon, and click **Open File** to view the output CSV file.

A custom model that detects text in a batch of images has now been deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/text-detection).