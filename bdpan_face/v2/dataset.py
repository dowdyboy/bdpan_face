import paddle.vision.transforms
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
import os
import cv2
import random
import numpy as np
from PIL import Image


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


class PairedMosaic():

    def __init__(self, prob=0.0, split_range=[0.25, 0.75], image_path_list=[], ):
        super().__init__()
        self.prob = prob
        self.split_range = split_range
        self.image_path_list = image_path_list
        self.image_index_list = list(range(len(self.image_path_list)))
        self.params = dict()

    def _get_params(self, inputs):
        params = {}
        params['mosic'] = random.random() < self.prob
        if params['mosic']:
            params['split_x'] = random.uniform(self.split_range[0], self.split_range[1])
            params['split_y'] = random.uniform(self.split_range[0], self.split_range[1])
            params['select_mosic_image_path'] = list(np.random.choice(self.image_index_list, 3, ))
        return params

    def _apply_image(self, image, image_gt):
        if self.params['mosic']:
            h, w, _ = image.shape
            split_x = int(w * self.params['split_x'])
            split_y = int(h * self.params['split_y'])
            image_x_list = [load_image(self.image_path_list[idx][0]) for idx in self.params['select_mosic_image_path']]
            image_gt_list = [load_image(self.image_path_list[idx][1]) for idx in self.params['select_mosic_image_path']]
            res_image = np.zeros((h, w, 3), dtype=np.uint8)
            res_gt = np.zeros((h, w, 3), dtype=np.uint8)
            res_image[:split_y, :split_x, :] = image[:split_y, :split_x, :]
            res_gt[:split_y, :split_x, :] = image_gt[:split_y, :split_x, :]
            res_image[:split_y, split_x:, :] = image_x_list[0][:split_y, split_x:, :]
            res_gt[:split_y, split_x:, :] = image_gt_list[0][:split_y, split_x:, :]
            res_image[split_y:, :split_x, :] = image_x_list[1][split_y:, :split_x, :]
            res_gt[split_y:, :split_x, :] = image_gt_list[1][split_y:, :split_x, :]
            res_image[split_y:, split_x:, :] = image_x_list[2][split_y:, split_x:, :]
            res_gt[split_y:, split_x:, :] = image_gt_list[2][split_y:, split_x:, :]
            return res_image, res_gt
        return image, image_gt

    def __call__(self, img_pair):
        image, image_gt = img_pair
        self.params = self._get_params(img_pair)
        image, image_gt = self._apply_image(image, image_gt)
        return image, image_gt


class FaceDataset(Dataset):

    def __init__(self, root_dir,
                 is_to_tensor=True,
                 use_cache=False,
                 is_train=True,
                 rate=1.0,
                 h_flip_p=0.5,
                 mosic_p=0.0, ):
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_to_tensor = is_to_tensor
        self.use_cache = use_cache
        self.is_train = is_train
        self.rate = rate
        self.image_path_list = []
        self.image_cache = dict()
        self.to_tensor = T.ToTensor()

        self._init_image_path()

        self.random_hflip = PairedRandomHorizontalFlip(prob=h_flip_p, keys=['image', 'image'], )
        self.mosaic = PairedMosaic(prob=mosic_p, image_path_list=self.image_path_list, )

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
        x, gt = self.mosaic((x, gt))
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
