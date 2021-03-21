from time import time
from typing import Any

from PIL import Image
from joblib import load

from src.utils.utils import ExifOrientation


def classify(image: Image):
    print('Classification initialized...')
    start_time = time()
    preprocess: ExifOrientation = ExifOrientation()
    image = preprocess(image)

    faces: Any = load('model/recognizer.pkl')(image)
    for face in faces:
        label, confidence = face.top_prediction.label.upper(), face.top_prediction.confidence * 100
        print(f'...recognized {label} with {round(confidence, 3)}% confidence')
    print(f'...completed in {round(time() - start_time, 2)} seconds')
