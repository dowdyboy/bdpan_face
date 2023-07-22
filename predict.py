import paddle
import cv2
import os
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T
# v1
# from bdpan_face.model import STRAIDR, STRAIDRLowPixel
# v2
from bdpan_face.v2.model import STRAIDR
import sys
import numpy as np


assert len(sys.argv) == 3
src_image_dir = sys.argv[1]
save_dir = sys.argv[2]
chk_path = 'checkpoints/v2/chk_best_step_1457500/model_0.pdparams'


def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(x, save_path):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, x)


def to_img_arr(x, un_norm=None):
    if un_norm is not None:
        y = un_norm((x, x, x))[0]
        y = y.numpy().transpose(1, 2, 0)
        y = np.clip(y, 0., 255.).astype(np.uint8)
    else:
        y = x.numpy().transpose(1, 2, 0)
        y = np.clip(y, 0., 1.)
        y = (y * 255).astype(np.uint8)
    return y


class FaceTestDataset(Dataset):

    def __init__(self, image_dir):
        super(FaceTestDataset, self).__init__()
        self.image_dir = image_dir
        self.image_list = []
        self.to_tensor = T.ToTensor()
        self._init_image_list()

    def _init_image_list(self):
        for filename in os.listdir(self.image_dir):
            self.image_list.append(
                os.path.join(self.image_dir, filename)
            )

    def __getitem__(self, idx):
        return self.to_tensor(load_image(self.image_list[idx])), self.image_list[idx]

    def __len__(self):
        return len(self.image_list)


def build_model():
    # baseline
    model = STRAIDR(unet_num_c=[16, 32, 64, 64, 128],
                    fine_num_c=[32],)
    # small
    # model = STRAIDR(unet_num_c=[8, 16, 32, 32, 64],
    #                 fine_num_c=[16])
    # lowpix
    # model = STRAIDRLowPixel(unet_num_c=[16, 32, 64, 64, 128],
    #                         fine_num_c=[32], )
    model.load_dict(paddle.load(chk_path))
    model.eval()
    return model


def build_data():
    dataset = FaceTestDataset(src_image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return loader, dataset


def main():
    model = build_model()
    test_loader, test_dataset = build_data()

    with paddle.no_grad():
        for bat_x, bat_path in test_loader:
            pred_y = model(bat_x)
            pred_im = to_img_arr(pred_y[0])
            save_image(pred_im, os.path.join(save_dir, os.path.basename(bat_path[0])))
    return


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main()
