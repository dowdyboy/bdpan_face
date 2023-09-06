import math

import paddle
from bdpan_face.v2.model import STRAIDR
from bdpan_face.v2.dataset import FaceDataset, load_image, save_image
import numpy
from PIL import Image
import cv2
from dowdyboy_lib.assist import get_conv_out_size
import sys

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

    # ds = FaceDataset(r'F:\BaiduNetdiskDownload\bdface_train_datasets',
    #                  is_to_tensor=False,
    #                  is_train=True,
    #                  mosic_p=1.0,)
    # for i in range(len(ds)):
    #     print(ds[i][0].shape, ds[i][1].shape)

    # a = load_image(r'E:\ideaworkspace4sota\I3D_Feature_Extraction_resnet-main\output\frames\01_0055_i3d\083.jpg')
    # b = load_image(r'E:\ideaworkspace4sota\I3D_Feature_Extraction_resnet-main\output\frames\01_0055_i3d\091.jpg')
    # cv2.calcOpticalFlowPyrLK(a, b)


    # from bdpan_face.v5.model import NAFNet
    # model = NAFNet(img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 12], dec_blk_nums=[1, 1, 1, 1])
    # x = paddle.rand((1, 3, 1024, 1024))
    # with paddle.no_grad():
    #     y = model(x)
    # paddle.save(model.state_dict(), 'tmp.pdparams')

    from bdpan_face.v4.model import STRAIDR
    model = STRAIDR(unet_num_c=[32, 64, 64, 64, 128],
                    fine_num_c=[128], )
    x = paddle.rand((1, 3, 1024, 1024))
    with paddle.no_grad():
        y = model(x)
    paddle.save(model.state_dict(), 'tmp.pdparams')

    print()


