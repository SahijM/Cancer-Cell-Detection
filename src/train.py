from model import unet_model
from data_preprocessing import load_and_preprocess, augment_data
from sklearn.model_selection import train_test_split

def train_model():
    print("Starting model training...")
    images, masks = load_and_preprocess()
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    model = unet_model()
    
    augmented_data = augment_data(X_train, y_train)
    
    history = model.fit(augmented_data,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        steps_per_epoch=len(X_train) // 32,
                        batch_size=32)
    
    model.save('unet_model.h5')
     
    print("Model training completed.")
    return history

if __name__ == "__main__":
    train_model()
