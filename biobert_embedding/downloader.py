import os
import tarfile
import requests
import tensorflow as tf
from pathlib import Path

dropbox_id = "https://www.dropbox.com/s/hvsemunmv0htmdk/biobert_v1.1_pubmed_pytorch_model.tar.gz?dl=0"
gdd_id = "1TFtdE5pu0LiFTD4p7NEESwyVbhrY2_04"

def download_file_from_google_drive(id, destination):

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True, verify = False)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True, verify = False)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_BioBert(location):

    model_path = Path.cwd()/'biobert_v1.1_pubmed_pytorch_model'

    if location == 'dropbox':
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            dataset = tf.keras.utils.get_file(fname=model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz",
            origin = dropbox_id)
            tar = tarfile.open(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
            tar.extractall()

        else:
            if not os.path.exists(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz"):
                dataset = tf.keras.utils.get_file(fname=model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz",
                origin=dropbox_id)
            if not os.path.exists(model_path/"pytorch_model.bin"):
                print("Extracting biobert model tar.gz")
                tar = tarfile.open(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
                tar.extractall(model_path)

    if location == 'google drive':
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            print("Downloading the biobert model, will take a minute...")
            download_file_from_google_drive(gdd_id, model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
            tar = tarfile.open(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
            tar.extractall(model_path)
        else:
            if not os.path.exists(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz"):
                download_file_from_google_drive(gdd_id, model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
            if not os.path.exists(model_path/"pytorch_model.bin"):
                print("Extracting biobert model tar.gz")
                tar = tarfile.open(model_path/"biobert_v1.1_pubmed_pytorch_model.tar.gz")
                tar.extractall(model_path)

    return model_path


if __name__ == "__main__":
    print("package from downloading biobert model")
