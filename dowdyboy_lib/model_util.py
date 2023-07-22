from torchinfo import summary
from ptflops.flops_counter import get_model_complexity_info
import torch
import os
import shutil


def frozen_module(module):
    for key, value in module.named_parameters():  # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
        value.requires_grad = False


def unfrozen_module(module):
    for key, value in module.named_parameters():
        value.requires_grad = True


def module_summary(model, in_size):
    in_size = (1, ) + in_size
    summary(model, input_size=in_size, device='cpu')


def module_cost(model, in_size):
    flops, params = get_model_complexity_info(model, in_size, as_strings=False, print_per_layer_stat=True)
    print('total params : ' + str(params / 1e6) + 'M')
    print('Flops: ' + str(flops / 1e9) + 'G')


def save_checkpoint(step, dir_path, model_list, optimizer_list=None, max_keep_num=10):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    sub_dir_names = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    sub_dir_names.sort(key=lambda x: os.path.getctime(os.path.join(dir_path, x)))
    sub_dir_names = list(filter(lambda x: x.startswith('chk_step_'), sub_dir_names))
    if len(sub_dir_names) >= max_keep_num:
        for sub_dir_name in sub_dir_names[:len(sub_dir_names) - max_keep_num + 1]:
            shutil.rmtree(os.path.join(dir_path, sub_dir_name))
    saved_dir_path = os.path.join(dir_path, f'chk_step_{step}')
    if not os.path.isdir(saved_dir_path):
        os.makedirs(saved_dir_path, exist_ok=True)
    for i, model in enumerate(model_list):
        torch.save(model.state_dict(), os.path.join(saved_dir_path, f'model_{i}.pt'))
    if optimizer_list is not None:
        for i, optimizer in enumerate(optimizer_list):
            torch.save(optimizer.state_dict(), os.path.join(saved_dir_path, f'optimizer_{i}.pt'))


def save_checkpoint_unique(step, dir_path, model_list, optimizer_list=None, label='unique'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    sub_dir_names = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    sub_dir_names = list(filter(lambda x: x.startswith(f'chk_{label}_'), sub_dir_names))
    for dir_name in sub_dir_names:
        shutil.rmtree(os.path.join(dir_path, dir_name))
    saved_dir_path = os.path.join(dir_path, f'chk_{label}_step_{step}')
    if not os.path.isdir(saved_dir_path):
        os.makedirs(saved_dir_path, exist_ok=True)
    for i, model in enumerate(model_list):
        torch.save(model.state_dict(), os.path.join(saved_dir_path, f'model_{i}.pt'))
    if optimizer_list is not None:
        for i, optimizer in enumerate(optimizer_list):
            torch.save(optimizer.state_dict(), os.path.join(saved_dir_path, f'optimizer_{i}.pt'))
