import numpy as np
from typing import List, Tuple


#########################################################################################
#                                       Transform                                       #
#########################################################################################


class Compose:
    def __init__(self, transforms: List['Transform']) -> None:
        self.transforms = transforms

    def __repr__(self) -> str:
        return f'Compose(transforms={self.transforms})'

    def __call__(self, img: np.ndarray or np.ndarray) -> np.ndarray or np.ndarray:
        for transform in self.transforms:
            img = transform(img)
        return img


class Transform:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.transform(img)

    def transform(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Resize(Transform):
    def __init__(self, size: Tuple[int, ...]) -> None:
        self.h, self.w = _pair(size)

    def __repr__(self) -> str:
        return f'Transform(Resize(size={self.size}))'

    def transform(self, img: np.ndarray) -> np.ndarray:
        c, h, w = img.shape
        img_rsz = np.zeros((c, self.h, self.w))
        for row in range(self.h):
            for col in range(self.w):
                img_rsz[:, row, col] = bilinear_interpolate(
                    img, (row / self.h * h, col / self.w * w)
                )

        return img_rsz


class CenterCrop(Transform):
    def __init__(self, size: Tuple[int, ...]) -> None:
        self.h, self.w = _pair(size)

    def __repr__(self) -> str:
        return f'Transform(CenterCrop(size={self.size}))'

    def transform(self, img: np.ndarray) -> np.ndarray:
        c, h, w = img.shape
        if self.h == h and self.w == w:
            return img
        elif self.h > h or self.w > w:
            resize_fn = Resize((self.h, self.w))
            return resize_fn(img)
        else:
            row0 = (h - self.h) // 2
            col0 = (w - self.w) // 2
            row1 = row0 + self.h
            col1 = col0 + self.w
            return img[:, row0:row1, col0:col1]


class RandomResizedCrop(Transform):
    def __init__(self, size: Tuple[int, ...]) -> None:
        self.h, self.w = _pair(size)

    def __repr__(self) -> str:
        return f'Transform(RandomResizedCrop(size={self.size}))'

    def transform(self, img: np.ndarray) -> np.ndarray:
        c, h, w = img.shape
        if self.h == h and self.w == w:
            return img
        elif self.h > h or self.w > w:
            resize_fn = Resize((self.h, self.w))
            return resize_fn(img)
        else:
            row0 = np.random.randint(0, h - self.h)
            col0 = np.random.randint(0, w - self.w)
            return img[:, row0 : row0 + self.h, col0 : col0 + self.w]


class RandomRotation(Transform):
    """Conter-clockwise rotation is positive, clockwise is negative."""

    def __init__(self, radians: Tuple[float, ...] = None) -> None:
        self.radians = radians if radians else (-np.pi / 12, np.pi / 12)

    def __repr__(self) -> str:
        return f'Transform(RandomRotation(radians={self.radians}))'

    def transform(self, img: np.ndarray) -> np.ndarray:
        c, h, w = img.shape
        theta = np.random.uniform(*self.radians)
        img_rot = np.zeros((c, h, w))

        center = (h // 2, w // 2)
        for row in range(h):
            for col in range(w):
                rho = np.sqrt((row - center[0]) ** 2 + (col - center[1]) ** 2)
                phi = np.arctan2(row - center[0], col - center[1]) + theta
                row_pre = rho * np.sin(phi) + center[0]
                col_pre = rho * np.cos(phi) + center[1]
                img_rot[:, row, col] = bilinear_interpolate(img, (row_pre, col_pre))

        return img_rot


class ToTensor(Transform):
    def __repr__(self) -> str:
        return 'Transform(ToTensor)'

    def transform(self, img: np.ndarray) -> np.ndarray:
        return img / 255.0


class Normalize(Transform):
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> None:
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return f'Transform(Normalize(mean={self.mean}, std={self.std}))'

    def transform(self, img: np.ndarray) -> np.ndarray:
        if np.isscalar(self.mean):
            mean = self.mean
        else:
            shape_ = [1 for _ in range(img.ndim)]
            shape_[0] = len(img) if len(self.mean) == 1 else len(self.mean)
            mean = np.reshape(self.mean, shape_)

        if np.isscalar(self.std):
            std = self.std
        else:
            shape_ = [1 for _ in range(img.ndim)]
            shape_[0] = len(img) if len(self.std) == 1 else len(self.std)
            std = np.reshape(self.std, shape_)

        return (img - mean) / std


#########################################################################################
#                                        Utility                                        #
#########################################################################################


def _pair(x):
    if np.isscalar(x):
        return x, x
    else:
        assert len(x) == 2, "x must be a scalar or a pair"
        return x


def bilinear_interpolate(img: np.ndarray, index: Tuple[float, float]) -> np.ndarray:
    h, w = img.shape[1:]
    row, col = index
    row0 = int(row)
    row1 = row0 + 1
    col0 = int(col)
    col1 = col0 + 1

    if row0 < 0 or row1 > h - 1 or col0 < 0 or col1 > w - 1:
        return 0

    row0_ratio = row - row0
    row1_ratio = 1 - row0_ratio
    col0_ratio = col - col0
    col1_ratio = 1 - col0_ratio

    return (
        img[:, row0, col0] * row0_ratio * col0_ratio
        + img[:, row0, col1] * row1_ratio * col0_ratio
        + img[:, row1, col0] * row0_ratio * col1_ratio
        + img[:, row1, col1] * row1_ratio * col1_ratio
    )
