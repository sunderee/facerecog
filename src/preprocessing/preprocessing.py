from os.path import abspath, curdir
from typing import Optional, List, Tuple, Dict

from PIL import Image
from numpy import ndarray
from numpy import stack
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize

from src.preprocessing.extractor import Extractor
from src.utils.utils import ExifOrientation


def load_data() -> Tuple[ndarray, List[int], Dict[str, int]]:
    transform: Compose = Compose([ExifOrientation(), Resize(1024)])
    dataset: ImageFolder = ImageFolder(f'{abspath(curdir)}/assets')

    processed_embeddings: List[ndarray] = []
    labels: List[int] = []

    for path, label in dataset.samples:
        embeddings: Optional[ndarray] = Extractor().extract_features(transform(Image.open(path).convert('RGB')))
        if embeddings is None:
            continue
        if embeddings.shape[0] > 1:
            embeddings = embeddings[0, :]
        processed_embeddings.append(embeddings.flatten())
        labels.append(label)

    return stack(processed_embeddings), labels, dataset.class_to_idx
