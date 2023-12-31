from pathlib import Path
import pickle

import numpy as np
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FFTModel:
    LOWPASS_FILTER = ImageFilter.Kernel((3, 3), (
            0, -1, 0,
            -1, 4, -1,
            0, -1, 0,
        ),
        1, 0
    )
    HIGHPASS_FILTER = ImageFilter.Kernel((5, 5), (
            1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1
        ),
        1/273, 0,
    )

    def __init__(self, model_gen: SVC | None = None):
        self._pcas: list[PCA] | None = None
        self._scaler: StandardScaler | None = None
        self._model: SVC = None  # type: ignore
        self._model_gen = model_gen
        self._id2label: dict = None  # type: ignore

    def _get_preprocessed_data(self, images: list[Image.Image]):
        data_fft = [
            np.fft.fft2(datum)  # type: ignore
            for datum in images
        ]
        data: list[list[np.ndarray]] = [[] for _ in range(4)]

        for i in range(len(images)):
            # Append each color channel to the corresponding list
            for j in range(3):
                data[j].append(np.abs(data_fft[i])[:, :, j].flatten())

            # Apply filters to each image and append to the edge list
            data[3].append(
                np.array(
                    images[i]
                    .convert("L")
                    .filter(self.LOWPASS_FILTER)
                    .filter(self.HIGHPASS_FILTER)
                ).flatten()
            )

        return data

    def _get_transformed_data(self, preprocessed_images, fit=False):
        if self._pcas is None or self._scaler is None:
            raise Exception("Model not trained")
        data = [
            self._pcas[i].fit_transform(datum) if fit
            else self._pcas[i].transform(datum)
            for i, datum in enumerate(preprocessed_images)
        ]
        data = np.concatenate((
            *data,
            (data[0] + data[1] + data[2] / 3),
        ), axis=1)
        if fit:
            return self._scaler.fit_transform(data)
        else:
            return self._scaler.transform(data)

    def fit(self, X: list[Image.Image], y, id2label=None):
        self._id2label = id2label
        d = self._get_preprocessed_data(X)
        pca_ncomp = [16, 16, 16, 2]
        self._pcas = [
            PCA(n_components=pca_ncomp[i])
            for i in range(len(d))
        ]
        self._scaler = StandardScaler()
        d = self._get_transformed_data(d, fit=True)
        if self._model_gen is None:
            self._model = SVC(kernel='rbf', C=4, random_state=42, probability=True, verbose=True)
        else:
            self._model = self._model_gen()
        return self._model.fit(d, y)

    def _preprocess_and_transform(self, X):
        if self._model is None or self._pcas is None or self._scaler is None:
            raise Exception("Model not trained")
        d = self._get_preprocessed_data(X)
        d = self._get_transformed_data(d, fit=False)
        return d

    def predict(self, X):
        return self._model.predict(self._preprocess_and_transform(X))

    def predict_proba(self, X, return_dict=False):
        p = self._model.predict_proba(self._preprocess_and_transform(X))
        if return_dict:
            p = [
                dict(sorted({
                    self._id2label[self._model.classes_[i]]: p[j][i]
                    for i in range(len(p[j]))
                }.items(), key=lambda x: x[1], reverse=True))
                for j in range(len(p))
            ]
        return p

    def single_predict(self, X):
        return self.predict_proba([X], return_dict=True)[0]

    def save(self, path):
        mod_gen = self._model_gen
        self._model_gen = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._model_gen = mod_gen

    @staticmethod
    def load(path) -> 'FFTModel':
        with open(path, "rb") as f:
            return pickle.load(f)


def get_image(path):
    return Image.open(path).resize((256, 256))


def get_image_data(paths, label2id: dict[str, int]):
    if isinstance(paths, str):
        paths = [paths]
    X = []
    y = []
    label_cnt = {k: 0 for k in label2id.keys()}
    for path in paths:
        p = Path(path)
        for label, id in label2id.items():
            new_parent = p / label
            if new_parent.exists():
                for file in new_parent.iterdir():
                    X.append(get_image(file))
                    y.append(id)
                    label_cnt[label] += 1
    for k, v in label_cnt.items():
        print(f"Count images for {k}: {v}")
    print("Total images:", sum(label_cnt.values()))
    return X, y
