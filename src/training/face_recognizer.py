from collections import namedtuple
from typing import Union, Any, List, Optional

from PIL import Image
from numpy import ndarray

from src.preprocessing.extractor import Extractor

Face = namedtuple('Face', 'top_prediction all_predictions')
Prediction = namedtuple('Prediction', 'label confidence')
BoundingBox = namedtuple('BoundingBox', 'left top right bottom')


class FaceRecognizer:
    def __init__(self, classifier: Union[type, Any], idx: dict):
        self.__classifier = classifier
        self.__idx_to_class = idx

    def __call__(self, image: Image) -> List[Face]:
        return self.recognize_faces(image)

    def recognize_faces(self, image: Image) -> List[Face]:
        embeddings: Optional[ndarray] = Extractor().extract_features(image)
        predictions: Any = self.__classifier.predict_proba(embeddings)

        return [
            Face(
                top_prediction=self.top_prediction(self.__idx_to_class, probability),
                all_predictions=self.__to_predictions(self.__idx_to_class, predictions)
            )
            for probability in predictions
        ]

    @staticmethod
    def top_prediction(idx_to_class: dict, probabilities: Any) -> Prediction:
        top_label: Any = probabilities.argmax()
        return Prediction(label=idx_to_class[top_label], confidence=probabilities[top_label])

    @staticmethod
    def __to_predictions(idx_to_class: dict, probabilities: Any) -> List[Prediction]:
        return [Prediction(label=idx_to_class[i], confidence=prob) for i, prob in enumerate(probabilities)]
