import paddle
from bdpan_face.v2.model import STRAIDR
from bdpan_face.dataset import FaceDataset
import numpy
from PIL import Image
import cv2
from dowdyboy_lib.assist import get_conv_out_size


if __name__ == '__main__':
    # model = STRAIDR(unet_num_c=[16, 32, 64, 64, 128],
    #                 fine_num_c=[32])
    # bat_x = paddle.rand((1, 3, 512, 512))
    # pred_y = model(bat_x)

    # dataset = FaceDataset(
    #     r'F:\BaiduNetdiskDownload\bdface_train_datasets',
    #     is_to_tensor=False,
    #     rate=0.01,
    #     h_flip_p=1.0,
    # )
    # idx = 0
    # for x, gt in dataset:
    #     idx += 1
    #     print()
    # print(idx)

    # model = STRAIDRLowPixel()
    # x = paddle.rand((1, 3, 1024, 1024))
    # y = model(x)


    # model = STRAIDR(unet_num_c=[16, 32, 64, 64, 128],
    #                 fine_num_c=[32])
    # x = paddle.rand((1, 3, 1024, 1024))
    # with paddle.no_grad():
    #     y = model(x)
    # paddle.save(model.state_dict(), 'tmp.pdparams')


    print()
