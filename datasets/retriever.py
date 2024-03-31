from pathlib import Path
from typing import Union
import os
import zipfile
import requests

def download_and_unzip_dataset_from_url(url: str, extract_to: str) -> None:
    """
    Download a zip file from a URL

    :param url: The URL containing the download zip.
    :param extract_to: The directory to extract the zip contents to.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    local_zip_path= os.path.join(extract_to, "dataset.zip")
    try: 
        with requests.get(url, stream=True) as r:
            with open(local_zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        unzip_dataset(local_zip_path, extract_to)
    finally:
        try:
            os.remove(local_zip_path)
        except FileNotFoundError:
            pass

def download_github_folder_contents(
    repo_owner: str,
    repo_name: str,
    folder_path: str,
    save_to: str
) -> None:
    """
    Download all files from a specific GitHub folder.

    :param repo_owner: The owner of the repository (user or organization).
    :param repo_name: The name of the repository.
    :param folder_path: The path to the folder inside the repository.
    :param save_to: Local directory to save the downloaded files.
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    response = requests.get(api_url)
    items = response.json()
    
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for item in items:
        if item["type"] == "file":
            download_url = item["download_url"]
            file_response = requests.get(download_url)
            file_path = os.path.join(save_to, os.path.basename(item["path"]))
            
            with open(file_path, 'wb') as file:
                file.write(file_response.content)
            print(f"Downloaded {file_path}")


def download_and_unzip_dataset_from_google_drive(file_id, extract_to):
    """
    Download a zip file from Google Drive

    :param file_id: The file id of the zip file in Google Drive.
    :param extract_to: The directory to extract the zip contents to.
    """
    try:
        URL = "https://drive.google.com/uc?export=download"
        CONFIRMATION_URL = "https://drive.usercontent.google.com/download"
        
        session = requests.Session()

        response = session.get(URL, params={ "id" : file_id }, stream=True)
        token = get_confirm_token(response)
        
        if token:
            params = { "id" : file_id, "confirm" : token }
            response = session.get(CONFIRMATION_URL, params=params, stream=True)
        
        save_response_content(response, "dataset.zip")
        unzip_dataset("dataset.zip", extract_to)
    finally:
        try:
            os.remove("dataset.zip")
        except FileNotFoundError:
            pass


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("NID"):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip_dataset(path: Union[Path, str], extract_to: str):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
