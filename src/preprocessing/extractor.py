from typing import Tuple, Optional

import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
from numpy import ndarray
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose

from src.utils.utils import Whitening


class Extractor:
    def __init__(self):
        self.__aligner: MTCNN = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9])
        self.__facenet_preprocessor: Compose = transforms.Compose([Whitening()])
        self.__facenet: InceptionResnetV1 = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_features(self, image: Image) -> Optional[ndarray]:
        detection_tuple: Tuple[ndarray, list] = self.__aligner.detect(image)
        if detection_tuple[0] is None:
            return None

        faces: Tensor = torch.stack([extract_face(image, box) for box in detection_tuple[0]])
        embeddings: ndarray = self.__facenet(self.__facenet_preprocessor(faces)).detach().numpy()
        return embeddings
