import os
import cv2
import base64
import numpy as np
import magic
import pathlib
import sys
import yaml
import requests

scripts_dir = pathlib.Path(__file__).parent.resolve()
easyocr_dir = os.path.join(scripts_dir, "easyocr")
sys.path.append(easyocr_dir)


BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/ocr/"

FILES = ["craft_mlt_25k.pth", "english_g2.pth"]


def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(
            current_dir + "/model_dir" + f"/{f}"
        ) and not os.path.exists("/input/train/" + f):
            print(f"Downloading file: {f}")
            response = requests.get(BASE_FOLDER_URL + f)
            with open(current_dir + "/model_dir/" + f, "wb") as fb:
                fb.write(response.content)


download_model_files()
import easyocr


# check if the user has set any parameters in the environment variables
if "lang_list" in os.environ:
    languages = os.environ["language"].split(",")
else:
    languages = ["en"]

if "decoder" in os.environ:
    decoder = os.environ["decoder"]
else:
    decoder = "wordbeamsearch"

if "beamWidth" in os.environ:
    beamWidth = int(os.environ["beamWidth"])
else:
    beamWidth = 10

if "contrast_ths" in os.environ:
    contrast_ths = os.environ["contrast_ths"]
else:
    contrast_ths = 0.1

if "adjust_contrast" in os.environ:
    adjust_contrast = os.environ["adjust_contrast"]
else:
    adjust_contrast = 0.5

if "text_threshold" in os.environ:
    text_threshold = os.environ["text_threshold"]
else:
    text_threshold = 0.5

if "link_threshold" in os.environ:
    link_threshold = os.environ["link_threshold"]
else:
    link_threshold = 0.5

if "mag_ratio" in os.environ:
    mag_ratio = os.environ["mag_ratio"]
else:
    mag_ratio = 1

if "height_ths" in os.environ:
    height_ths = os.environ["height_ths"]
else:
    height_ths = 0.5

if "width_ths" in os.environ:
    width_ths = os.environ["width_ths"]
else:
    width_ths = 0.5

reader = 0
# check if the path exists for /input/ocr_train/custom_model.pth which means the user has retrained a model
if os.path.exists("/input/train/custom_model.pth"):

    with open("/input/train/custom_model.yaml", "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    reader = easyocr.Reader(
        [opt["lang_list"][0]],
        model_storage_directory="/input/train",
        user_network_directory="/input/train",
        recog_network="custom_model",
        download_enabled=False,
    )
    decoder = "greedy"

# else load default model
else:

    reader = easyocr.Reader(
        languages,
        model_storage_directory=os.path.join(scripts_dir, "model_dir"),
        download_enabled=False,
    )


def predict(data):
    output = {}

    for image_number, image_data in enumerate(data["img"]):
        output[image_number + 1] = []
        decoded = base64.b64decode(image_data)  # decode the input image
        # figure out the image extension like is it .png or .jpg etc
        file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
        savepath = f"img.{file_ext}"
        nparr = np.fromstring(decoded, np.uint8)
        img_dec = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(savepath, img_dec)  # save the decoded image
        result = reader.readtext(  # run the OCR model with the loaded paramters
            savepath,
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            decoder=decoder,
            beamWidth=beamWidth,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            height_ths=height_ths,
            width_ths=width_ths,
            mag_ratio=mag_ratio,
        )
        for i in range(len(result)):
            x = result[i][0][0][0]
            y = result[i][0][0][1]
            w = result[i][0][1][0] - result[i][0][0][0]
            h = result[i][0][2][1] - result[i][0][1][1]
            response = {}
            response = {
                "text": result[i][1],
                "bounding_box": [int((x + w) / 2), int((y + h) / 2), int(w), int(h)],
                "confidence": result[i][-1],
            }
            output[image_number + 1].append(response)

    return {"prediction": output}
