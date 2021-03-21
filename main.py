from argparse import ArgumentParser, Namespace
from os.path import abspath, curdir

from PIL import Image

from src.classify.classify import classify
from src.training.train import train


def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser(description='facerecog is a facial recognition system developed in Python')
    parser.add_argument(
        '-t',
        '--only-train',
        help='Whether or not the program should run training stage',
        action='store_true'
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='Absolute path to the image used for testing'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.only_train:
        train()
    else:
        image_path = args.dir
        if image_path is not None:
            classify(Image.open(image_path).convert('RGB'))
        else:
            print('You need to specify test image directory')
