import paddle
import os
import shutil

@paddle.no_grad()
def print_model(model, input_size, custom_ops=None, ):
    def number_format(x):
        return f'{round(x / 1.e6, 2)}M, {round(x / 1.e9, 2)}G'
    summary_res = paddle.summary(model, input_size, )
    flops_res = paddle.flops(model, input_size, custom_ops=custom_ops, )
    total_params = summary_res['total_params']
    trainable_params = summary_res['trainable_params']
    flops_count = flops_res
    print('=========================================================')
    print(f'total params: {number_format(total_params)}, '
          f'trainable params: {number_format(trainable_params)}, '
          f'flops: {number_format(flops_count)}')
    bat_x = paddle.rand(input_size)
    pred_y = model(bat_x)
    if isinstance(pred_y, tuple) or isinstance(pred_y, list):
        print(f'input shape: {bat_x.shape}, output shape: {[y.shape for y in pred_y]}')
    else:
        print(f'input shape: {bat_x.shape}, output shape: {pred_y.shape}')


def frozen_layer(layer):
    for v in layer.parameters():
        v.trainable = False


def unfrozen_layer(layer):
    for v in layer.parameters():
        v.trainable = True

# def init_model(model):
#     def reset_func(m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             constant_(m.bias, 0)
#             constant_(m.weight, 1.0)
#         elif hasattr(m, 'weight') and (not isinstance(
#                 m, (nn.BatchNorm, nn.BatchNorm2D))):
#             kaiming_uniform_(m.weight, a=math.sqrt(5))
#             if m.bias is not None:
#                 fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
#                 bound = 1 / math.sqrt(fan_in)
#                 uniform_(m.bias, -bound, bound)
#     model.apply(reset_func)


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
        paddle.save(model.state_dict(), os.path.join(saved_dir_path, f'model_{i}.pdparams'))
    if optimizer_list is not None:
        for i, optimizer in enumerate(optimizer_list):
            paddle.save(optimizer.state_dict(), os.path.join(saved_dir_path, f'optimizer_{i}.pdopt'))


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
        paddle.save(model.state_dict(), os.path.join(saved_dir_path, f'model_{i}.pdparams'))
    if optimizer_list is not None:
        for i, optimizer in enumerate(optimizer_list):
            paddle.save(optimizer.state_dict(), os.path.join(saved_dir_path, f'optimizer_{i}.pdopt'))

