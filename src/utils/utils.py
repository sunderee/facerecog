from typing import List, Any, Union

from PIL.Image import FLIP_LEFT_RIGHT, ROTATE_180, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_270, Image
from torch import Tensor


class Whitening(object):
    def __call__(self, image: Tensor) -> Tensor:
        mean: Tensor = image.mean()
        return (image - mean) / image.std().clamp(min=1.0 / (float(image.numel())) ** 0.5)


class ExifOrientation(object):
    __exif_orientation_tag: int = 0x0112
    __exif_transpose_sequences: List[Union[List[Any], List[int]]] = [[], [], [FLIP_LEFT_RIGHT], [ROTATE_180],
                                                                     [FLIP_TOP_BOTTOM],
                                                                     [FLIP_LEFT_RIGHT, ROTATE_90], [ROTATE_270],
                                                                     [FLIP_TOP_BOTTOM, ROTATE_90],
                                                                     [ROTATE_90], ]

    def __call__(self, image: Image) -> Image:
        if 'parsed_exif' in image.info and self.__exif_orientation_tag in image.info['parsed_exif']:
            orientation: int = image.info['parsed_exif'][self.__exif_orientation_tag]
            transposes: Union[List[Any], List[int]] = self.__exif_transpose_sequences[orientation]
            for trans in transposes:
                image = image.transpose(trans)
        return image
