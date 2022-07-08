import requests
import os
import pathlib
import shutil
import tarfile
BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/ocr/"

FILES = ['dict.tar.gz']


def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(current_dir+f'/{f}') and not os.path.exists('/input/train/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            with open(f, "wb") as fb:
                fb.write(response.content)
            #shutil.move(current_dir+f'/{f}',current_dir+'/model_dir')
            file = tarfile.open('dict.tar.gz')
            file.extractall('./')
            file.close()
            shutil.move('dict',current_dir+'/easyocr')

