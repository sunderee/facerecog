from os import mkdir
from os.path import join, isdir
from time import time
from typing import Any, Union

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.preprocessing.preprocessing import load_data
from src.training.face_recognizer import FaceRecognizer

MODEL_DIR_PATH = 'model'


def train():
    print('Training the model...')
    start_time = time()
    embeddings, labels, class_to_idx = load_data()

    softmax: LogisticRegression = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    grid_search_cv: GridSearchCV = GridSearchCV(estimator=softmax,
                                                param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, cv=3)
    grid_search_cv.fit(embeddings, labels)
    clf: Union[type, Any] = grid_search_cv.best_estimator_

    idx_to_class: dict = {v: k for k, v in class_to_idx.items()}

    if not isdir(MODEL_DIR_PATH):
        mkdir(MODEL_DIR_PATH)
    model_path: str = join('model', 'recognizer.pkl')
    dump(FaceRecognizer(clf, idx_to_class), model_path)
    print(f'...completed in {round(time() - start_time, 2)} seconds')
