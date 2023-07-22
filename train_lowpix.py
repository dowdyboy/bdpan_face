import argparse
import os
import paddle
import numpy as np
import random
from PIL import Image
from bdpan_face.model import STRAIDRLowPixel
from bdpan_face.dataset import FaceDataset
from bdpan_face.optim import CosineAnnealingRestartLR
from paddle.io import DataLoader
import dowdyboy_lib.log as logger
from dowdyboy_lib.paddle.model_util import save_checkpoint, save_checkpoint_unique
from bdpan_face.psnr_ssim import calculate_psnr, calculate_ssim
from bdpan_face.init import init_model


parser = argparse.ArgumentParser(description='train shuiyin multi scale image')
# data config
parser.add_argument('--data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--num-workers', type=int, default=0, help='num workers')
# optimizer config
parser.add_argument('--lr', type=float, default=2e-4, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=0., help='model weight decay')
# train config
parser.add_argument('--iter', type=int, default=100, help='epoch num')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--seed', type=int, default=831, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='log process')
parser.add_argument('--val-interval', type=int, default=3000, help='log process')
parser.add_argument('--save-interval', type=int, default=3000, help='log process')
parser.add_argument('--resume', type=str, default=None, help='resume model')
parser.add_argument('--resume-iter', type=int, default=-1, help='resume iter')
args = parser.parse_args()


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def build_model():
    model = STRAIDRLowPixel(unet_num_c=[16, 32, 64, 64, 128],
                            fine_num_c=[32], )
    init_model(model)
    return model


def build_optimizer(model):
    interval = 500000
    # interval = 110000
    lr = args.lr
    lr_scheduler = None
    if args.use_scheduler:
        # lr = paddle.optimizer.lr.CosineAnnealingDecay(lr, args.epoch, last_epoch=args.last_epoch, verbose=True)
        lr = CosineAnnealingRestartLR(
            lr,
            periods=[interval, interval, interval, interval],
            restart_weights=[1, 1, 1, 1],
            eta_min=args.lr * 0.01,
            last_epoch=args.resume_iter,
        )
        lr_scheduler = lr
    if args.use_warmup:
        lr = paddle.optimizer.lr.LinearWarmup(lr, 10, args.lr * 0.1, args.lr, last_epoch=args.resume_iter, verbose=True)
        lr_scheduler = lr
    optimizer = paddle.optimizer.Adam(
        lr,
        parameters=[{
            'params': m.parameters()
        } for m in model] if isinstance(model, list) else model.parameters(),
        weight_decay=args.weight_decay,
        beta1=0.9,
        beta2=0.99,
    )
    return optimizer, lr_scheduler


def build_data():
    train_dataset = FaceDataset(root_dir=args.data_dir, is_train=True, rate=0.99, h_flip_p=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, )
    val_dataset = FaceDataset(root_dir=args.data_dir, is_train=False, rate=0.99)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers, drop_last=False, )
    return train_loader, val_loader, train_dataset, val_dataset


def update_iter(d_train_iter,
                d_train_loader, d_step):
    if d_train_iter is None or (d_step - 1) % len(d_train_loader) == 0:
        d_train_iter = iter(d_train_loader)
    return d_train_iter


def load_checkpoint(resume_dir, model_list, optimizer_list, ):
    model_chk_names = list(sorted(list(filter(lambda x: x.startswith('model'), os.listdir(resume_dir)))))
    optimizer_chk_names = list(sorted(list(filter(lambda x: x.startswith('optimizer'), os.listdir(resume_dir)))))
    for idx, filename in enumerate(model_chk_names):
        model_list[idx].set_state_dict(paddle.load(os.path.join(resume_dir, filename)))
    for idx, filename in enumerate(optimizer_chk_names):
        optimizer_list[idx].set_state_dict(paddle.load(os.path.join(resume_dir, filename)))


if __name__ == '__main__':
    out_dir = args.out_dir
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    best_psnr = -1
    start_step = 1 if args.resume_iter == -1 else args.resume_iter + 1

    set_seed(args.seed)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    logger.logging_conf(os.path.join(out_dir, 'runtime.log'))
    logger.log(args)

    train_loader, val_loader, train_dataset, val_dataset = build_data()
    logger.log(f'train size : {len(train_dataset)} , val size : {len(val_dataset)}')
    train_iter = None
    model = build_model()
    optimizer, lr_scheduler = build_optimizer(model)
    loss_func = paddle.nn.L1Loss()

    if args.resume is not None:
        load_checkpoint(args.resume, [model], [optimizer], )
        logger.log(f'resume from {args.resume}')

    model.train()
    for step in range(start_step, args.iter + 1):
        train_iter = update_iter(train_iter, train_loader, step)
        bat_x, bat_y = next(train_iter)

        optimizer.clear_grad()
        pred_y = model(bat_x)
        loss = loss_func(pred_y, bat_y)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            logger.log(f'step {step} loss : {loss.item()}, lr : {optimizer.get_lr()}')

        if step % args.save_interval == 0:
            save_checkpoint(step, checkpoint_dir, [model], [optimizer], max_keep_num=100)

        if step % args.val_interval == 0:
            psnr_list = []
            loss_list = []
            model.eval()
            with paddle.no_grad():
                for val_step, (val_x, val_y) in enumerate(val_loader):
                    val_pred_y = model(val_x)
                    val_loss = loss_func(val_pred_y, val_y)
                    val_pred_im = to_img_arr(val_pred_y[0])
                    val_im = to_img_arr(val_y[0])
                    val_input_im = to_img_arr(val_x[0])
                    psnr = float(calculate_psnr(val_pred_im, val_im, crop_border=4, test_y_channel=True, ))
                    psnr_list.append(psnr)
                    loss_list.append(val_loss.item())
                    Image.fromarray(val_pred_im).save(os.path.join(out_dir, f'{val_step}_pred.jpg'))
                    Image.fromarray(val_im).save(os.path.join(out_dir, f'{val_step}_gt.jpg'))
                    Image.fromarray(val_input_im).save(os.path.join(out_dir, f'{val_step}_input.jpg'))
            mean_psnr = float(np.mean(np.array(psnr_list)))
            mean_loss = float(np.mean(np.array(loss_list)))
            logger.log(f'step : {step} , mean psnr : {mean_psnr} , mean loss : {mean_loss}')
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                save_checkpoint_unique(step, checkpoint_dir, [model], [optimizer], label='best')
            model.train()

        if lr_scheduler is not None:
            lr_scheduler.step()

    print()


