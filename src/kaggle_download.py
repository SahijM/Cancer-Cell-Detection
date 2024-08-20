import os
from zipfile import ZipFile

def download_kaggle_dataset():
    kaggle_dataset = "data-science-bowl-2018"
    download_path = "./data/raw/"
    image_dir = "./data/images/"
    mask_dir = "./data/masks/"

    os.makedirs(download_path, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle competitions download -c {kaggle_dataset} -p {download_path}")

    print("Extracting dataset...")
    for filename in os.listdir(download_path):
        if filename.endswith(".zip"):
            with ZipFile(os.path.join(download_path, filename), 'r') as zip_ref:
                zip_ref.extractall(download_path)

    print("Organizing dataset...")
    for root, _, files in os.walk(download_path):
        for file in files:
            if file.endswith(".png"):
                if "mask" in file:
                    os.rename(os.path.join(root, file), os.path.join(mask_dir, file))
                else:
                    os.rename(os.path.join(root, file), os.path.join(image_dir, file))

    print("Dataset download and extraction complete.")

if __name__ == "__main__":
    download_kaggle_dataset()
