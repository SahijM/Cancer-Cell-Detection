from tensorflow.keras.models import load_model # type: ignore
from data_preprocessing import load_and_preprocess
from sklearn.metrics import jaccard_score
import numpy as np

def evaluate_model():
    model = load_model('unet_model.h5')
    images, masks = load_and_preprocess()
    
    preds = model.predict(images)
    preds = np.round(preds)
    
    iou_scores = [jaccard_score(masks[i].flatten(), preds[i].flatten()) for i in range(len(masks))]
    
    mean_iou = np.mean(iou_scores)
    print(f'Mean IoU: {mean_iou}')
    
if __name__ == "__main__":
    evaluate_model()
