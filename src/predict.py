import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from skimage.io import imread
import cv2

def predict_image(image_path):
    model = load_model('unet_model.h5')
    img = imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    
    pred = model.predict(img)
    pred = pred[0, :, :, 0]
    
    plt.imshow(pred, cmap='gray')
    plt.show()

if __name__ == "__main__":
    predict_image('./src/data/images/sample_image.png')
