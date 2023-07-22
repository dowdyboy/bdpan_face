import paddle.vision.transforms
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
import os
import cv2
import random


def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(x, save_path):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, x)


class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.hflip(v) for v in image]
            else:
                return F.hflip(image)
        return image


class FaceDataset(Dataset):

    def __init__(self, root_dir,
                 is_to_tensor=True,
                 use_cache=False,
                 is_train=True,
                 rate=1.0,
                 h_flip_p=0.5):
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_to_tensor = is_to_tensor
        self.use_cache = use_cache
        self.is_train = is_train
        self.rate = rate
        self.image_path_list = []
        self.image_cache = dict()
        self.to_tensor = T.ToTensor()

        self.random_hflip = PairedRandomHorizontalFlip(prob=h_flip_p, keys=['image', 'image'], )

        self._init_image_path()

    def _init_image_path(self):
        x_dir = os.path.join(self.root_dir, 'image')
        gt_dir = os.path.join(self.root_dir, 'groundtruth')
        for file_name in os.listdir(x_dir):
            self.image_path_list.append([
                os.path.join(x_dir, file_name),
                os.path.join(gt_dir, file_name)
            ])
        if self.is_train:
            self.image_path_list = self.image_path_list[:int(len(self.image_path_list) * self.rate)]
        else:
            self.image_path_list = self.image_path_list[int(len(self.image_path_list) * self.rate):]

    def _load_image(self, filepath):
        if self.use_cache:
            if filepath in self.image_cache.keys():
                return self.image_cache[filepath]
            else:
                img = load_image(filepath)
                self.image_cache[filepath] = img
                return img
        else:
            # img = load_image(filepath)
            # img = paddle.vision.transforms.resize(img, (256, 256))
            # return img
            return load_image(filepath)

    def _apply_aug(self, x, gt):
        x, gt = self.random_hflip((x, gt))
        return x, gt

    def __getitem__(self, idx):
        x = self._load_image(self.image_path_list[idx][0])
        gt = self._load_image(self.image_path_list[idx][1])

        x, gt = self._apply_aug(x, gt)

        if self.is_to_tensor:
            x = self.to_tensor(x)
            gt = self.to_tensor(gt)
        return x, gt

    def __len__(self):
        return len(self.image_path_list)
