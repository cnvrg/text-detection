import os
import cv2
import pandas as pd
import argparse
import pathlib
import sys

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
scripts_dir = pathlib.Path(__file__).parent.resolve()
easyocr_dir = os.path.join(scripts_dir, "easyocr")
sys.path.append(easyocr_dir)

import easyocr

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "--img_dir",
    action="store",
    dest="img_dir",
    default="/data/ocr_data/",
    required=True,
    help="""ocr dataset""",
)
parser.add_argument(
    "--lang_list",
    action="store",
    dest="lang_list",
    default="en",
    required=False,
    help="""language of the text inside the images""",
)
parser.add_argument(
    "--decoder",
    action="store",
    dest="decoder",
    default="wordbeamsearch",
    required=False,
    help="""decoder algorithm to parse text""",
)
parser.add_argument(
    "--beamWidth",
    action="store",
    dest="beamWidth",
    default="5",
    required=False,
    help="""How many beam to keep when decoder = ‘beamsearch’ or ‘wordbeamsearch’""",
)
parser.add_argument(
    "--contrast_ths",
    action="store",
    dest="contrast_ths",
    default="0.1",
    required=False,
    help="""Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to ‘adjust_contrast’ value. The one with more confident level will be returned as a result""",
)
parser.add_argument(
    "--adjust_contrast",
    action="store",
    dest="adjust_contrast",
    default="0.5",
    required=False,
    help="""target contrast level for low contrast text box""",
)
parser.add_argument(
    "--text_threshold",
    action="store",
    dest="text_threshold",
    default="0.5",
    required=False,
    help="""refers to the certainity required for a something to be classified as a letter. The higher this value the clearer characters need to look""",
)
parser.add_argument(
    "--link_threshold",
    action="store",
    dest="link_threshold",
    default="0.5",
    required=False,
    help="""amount of distance allowed between two characters for them to be seen as a single word. Higher the distance, higher probability of different sentences to be classified as a single sentence.""",
)
parser.add_argument(
    "--mag_ratio",
    action="store",
    dest="mag_ratio",
    default="1",
    required=False,
    help="""refers to the image magnification ratio""",
)
parser.add_argument(
    "--height_ths",
    action="store",
    dest="height_ths",
    default="0.5",
    required=False,
    help="""refers to the Maximum different in box height. Boxes with very different text size should not be merged.""",
)
parser.add_argument(
    "--width_ths",
    action="store",
    dest="width_ths",
    default="0.5",
    required=False,
    help="""Maximum horizontal distance to merge boxes""",
)

args = parser.parse_args()
lang_list = args.lang_list
img_dir = args.img_dir
decoder = args.decoder
beamWidth = int(args.beamWidth)
contrast_ths = float(args.contrast_ths)
adjust_contrast = float(args.adjust_contrast)
text_threshold = float(args.text_threshold)
link_threshold = float(args.link_threshold)
mag_ratio = float(args.mag_ratio)
height_ths = float(args.height_ths)
width_ths = float(args.width_ths)


lang_lis = lang_list.split(",")
count_1 = 0
reader = easyocr.Reader(lang_lis)  # load the language model into memory
df = pd.DataFrame(
    columns=["filename", "text", "x_coord", "y_coord", "width", "height", "confidence"]
)
for files in os.listdir(img_dir):
    # we only read files ending with specific extensions to avoid any errors due to reading non image files
    if (
        files.endswith("jpg")
        | files.endswith("png")
        | files.endswith("PNG")
        | files.endswith("jpeg")
    ):
        files1 = os.path.join(img_dir, files)
        img = cv2.imread(files1)  # read the file
        result = reader.readtext(  # run the ocr model
            files1,
            rotation_info=[90, 180, 270],
            contrast_ths=contrast_ths,
            adjust_contrast=adjust_contrast,
            decoder=decoder,
            beamWidth=beamWidth,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            height_ths=height_ths,
            width_ths=width_ths,
        )
        for i in range(len(result)):
            x = result[i][0][0][0]
            y = result[i][0][0][1]
            w = result[i][0][1][0] - result[i][0][0][0]
            h = result[i][0][2][1] - result[i][0][1][1]
            cv2.rectangle(  # draw boxes on the detected text
                img,
                (int(x), int(y)),
                (int(x) + int(w), int(y) + int(h)),
                (0, 255, 0),
                2,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(x), int(y))
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2
            img = cv2.putText(  # write text on the image
                img, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA
            )
            count_1 = count_1 + 1
            df.at[count_1, "filename"] = files
            df.at[count_1, "text"] = result[i][1]
            df.at[count_1, "x_coord"] = (x + w) / 2
            df.at[count_1, "y_coord"] = (y + h) / 2
            df.at[count_1, "width"] = w
            df.at[count_1, "height"] = h
            df.at[count_1, "confidence"] = result[i][-1]
        cv2.imwrite(cnvrg_workdir + files, img)
df.to_csv(cnvrg_workdir + "/output.csv")  # save the results to a csv file
