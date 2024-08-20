import os
import cv2
import numpy as np
from skimage.io import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from kaggle_download import download_kaggle_dataset

# Path to images and masks
RAW_IMAGE_DIR = './src/images/'
RAW_MASK_DIR = './src/masks/'
PROCESSED_DIR = './src/processed/'

print(os.listdir(RAW_IMAGE_DIR))
print(os.listdir(RAW_MASK_DIR))


def load_and_preprocess():
    print("Loading and preprocessing data...")
    
    if not os.listdir(RAW_IMAGE_DIR) or not os.listdir(RAW_MASK_DIR):
        print("No images found. Downloading dataset...")
        download_kaggle_dataset()

    images = []
    masks = []
    
    print("Processing images...")
    for filename in os.listdir(RAW_IMAGE_DIR):
        img = imread(os.path.join(RAW_IMAGE_DIR, filename))
        img = cv2.resize(img, (128, 128))
        images.append(img)
        
        mask = imread(os.path.join(RAW_MASK_DIR, filename.replace('.png', '_mask.png')))
        mask = cv2.resize(mask, (128, 128))
        masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    
    # Normalizing images and masks
    images = images / 255.0
    masks = masks / 255.0
    
    print(f"Processed {len(images)} images and {len(masks)} masks.")
    
    return images, masks

def augment_data(images, masks):
    print("Augmenting data...")
    
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_generator = image_datagen.flow(images, batch_size=32, seed=seed)
    mask_generator = mask_datagen.flow(masks, batch_size=32, seed=seed)

    return zip(image_generator, mask_generator)

if __name__ == "__main__":
    load_and_preprocess()
    print("Preprocessing complete.")
