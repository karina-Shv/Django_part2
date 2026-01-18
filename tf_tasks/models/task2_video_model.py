import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions, MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

video_model = MobileNetV2(weights="imagenet")

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.resize(frame, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = video_model.predict(x)
        decoded = decode_predictions(preds, top=1)[0][0][1]
        results.append(decoded)
        
    cap.release()
    if results:
        return max(set(results), key=results.count)
    return "Нічого не знайдено"