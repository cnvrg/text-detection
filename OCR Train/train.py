from easyocr import easyocr
import os
import shutil
import torch.backends.cudnn as cudnn
import yaml
import pandas as pd
import argparse
import pathlib
import sys
from run_trainer import train
from utils import AttrDict
from sklearn.model_selection import train_test_split
from prerun import download_model_files
download_model_files()
cudnn.benchmark = True
cudnn.deterministic = False
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
parser = argparse.ArgumentParser(
    description="""Retrain OCR model on a particular language"""
)
parser.add_argument(
    "-l",
    "--language",
    default="en",
    required=True,
    help="""Language code of the model you want to retrain""",
)
parser.add_argument(
    "--iterations",
    default="300000",
    help="""Number of iterations for training the model""",
    type=int,
)
parser.add_argument(
    "--data",
    default="",
    help="""Location of the folder containing train and val data.""",
)
args = parser.parse_args()


language = args.language
iterations = args.iterations
data = args.data

####### Train Val split ###########
labels = pd.read_csv(data + "/labels.csv")
file_name = labels["filename"]
words = labels["words"]
train_files, val_files, train_words, val_words = train_test_split(
    file_name, words, test_size=0.3, random_state=1
)
os.mkdir(data + "/train")
os.mkdir(data + "/val")
# save train csv in train folder
train_labels = {"filename": train_files, "words": train_words}
traindatalabels = pd.DataFrame.from_dict(train_labels).to_csv(
    data + "/train/labels.csv", index=False
)

# save val csv in val folder
val_labels = {"filename": val_files, "words": val_words}
valdatalabels = pd.DataFrame.from_dict(val_labels).to_csv(
    data + "/val/labels.csv", index=False
)

#function to move files iteratively form one folder to another
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(data + "/" + f, destination_folder)
        except:
            print(f)
            assert False


# move train images in train folder
move_files_to_folder(train_files, data + "/train")
# move val images in val folder
move_files_to_folder(val_files, data + "/val")

curr_path = pathlib.Path(__file__).parent.resolve()
reader = easyocr.Reader( #easyocr will download the relevant language model
    [language],
    model_storage_directory=cnvrg_workdir,  
)


model_train = "" 
for model_downloaded in os.listdir(cnvrg_workdir): #set model_train object to point to the language model downloaded
    if model_downloaded.endswith(".pth") and not model_downloaded.startswith("craft"):
        model_train = model_downloaded

#configure the training yaml
def get_config(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    os.makedirs(cnvrg_workdir + f"/{opt.experiment_name}", exist_ok=True)
    opt.num_iter = iterations
    opt.valInterval = (opt.num_iter) // 10
    opt.valInterval = max(1, opt.valInterval)
    opt.saved_model = cnvrg_workdir+"/" + model_train

    return opt


opt = get_config(os.path.join(curr_path, "config_files/train_config.yaml"))
lang_char = ""
#load the character file present from language dictionary files to read all the characters in the language
for character_file in os.listdir(os.path.join(curr_path, "easyocr/character")):
    if character_file.startswith(language):
        with open(
            os.path.join(curr_path, "easyocr/character", character_file),
            encoding="utf-8",
        ) as f:
            char = f.readline()
            while len(char) != 0:
                lang_char = lang_char + (char.rstrip("\n"))
                char = f.readline()
opt.lang_char = lang_char
opt.character = opt.number + opt.symbol + opt.lang_char
opt.train_data = data
opt.valid_data = data + "/val"

os.chdir(curr_path)

# create the network file
networkfile = "custom_model.yaml"
default_file = open(networkfile, "r")
custom_file = open(cnvrg_workdir + "/custom_model.yaml", "w")
for i in range(6):
    custom_file.write(default_file.readline())
# update the language here
custom_file.write(default_file.readline().replace("en", language))
# update the character list here
characters = "character_list: " + opt.character
custom_file.write(characters)
default_file.close()
custom_file.close()
# move custom_model.py to /cnvrg
shutil.move("custom_model.py", cnvrg_workdir)
train(opt, amp=False)
