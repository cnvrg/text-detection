You can use this blueprint to predict the text in a batch of images. In order to use this blueprint, you would need to provide one folder located in s3:

img_dir: A folder with all the images you want to predict the text on
Click on Use Blueprint button

You will be redirected to your blueprint flow page

In the flow, edit the following tasks to provide your data:

In the easy_ocr_batch task
Under the bucketname parameter provide the bucket name of the data
Under the prefix parameter provide the main path to where the images folder is located

NOTE: You can use prebuilt data examples paths that are already provided

Click on the 'Run Flow' button
In a few minutes you will predict the text in a batch of images and download the CSV file with the information about the text and bounding boxes.
Go to output artifacts and check for the bounding box images as well as the output csv file.  
You can use the Try it Live section with any image to check your model.  
You can also integrate your API with your code using the integration panel at the bottom of the page

[See here how we created this blueprint.](https://github.com/cnvrg/text-detection)