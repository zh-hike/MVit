from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import os
import paddle
import torch

_feats = []


def convert_layer_name(s):
    s = s + '.'
    m = s.split('.')
    new_s = []
    for i in range(len(m)-1):
        try:
            int(m[i])
            new_s.pop()
            new_s.append('[')
            new_s.append(m[i])
            new_s.append('].')
        except:
            new_s.append(m[i])
            new_s.append('.')
    return ''.join(new_s).strip('.')

def get_layer_name_paddle(net):
    names = []
    for key, v in net.named_sublayers():
        if '.' in key and key.split('.')[-1]!='blocks':
            names.append(convert_layer_name(key))
    return names

def get_layer_name_torch(net):
    names = []
    for key, v in net.named_modules():
        if '.' in key and key.split('.')[-1]!='blocks':
            names.append(convert_layer_name(key))
    return names


def _hook(module, inputs, output):
    _feats.append(output.detach().cpu().numpy())


def _register_forward_post_hook_paddle(names, model):
    handles = []
    for name in names:
        if '.' in name and name.split('.')[-1]!='blocks':
            h = f'model.{name}'
            handle = eval(h).register_forward_post_hook(_hook)
            handles.append(handle)
    return handles

def _register_forward_post_hook_torch(names, model):
    handles = []
    for name in names:
        if ('.' in name and name.split('.')[-1]!='blocks'):
            h = f'model.{name}'
            handle = eval(h).register_forward_hook(_hook)
            handles.append(handle)
    return handles

def add_log(log, names):
    assert len(names) == len(_feats), f"length of names is {len(names)}, bug length of _feats is {len(_feats)}"
    for name, feat in zip(names, _feats):
        if '.' in name and name.split('.')[-1]!='blocks':
            log.add(name, feat)


def clean_handles(handles):
    for i in range(len(handles)):
        handles[i].remove()

def forward(inputs, paddle_model, torch_model, save_path='./align', torch_features=False):
    os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
    paddle_inputs = None
    torch_inputs = None
    if isinstance(inputs, (paddle.Tensor, torch.Tensor)):
        inputs = inputs.detach().cpu().numpy()
    paddle_inputs = paddle.to_tensor(inputs)
    torch_inputs = torch.from_numpy(inputs)

    paddle_log = ReprodLogger()

    paddle_names = get_layer_name_paddle(paddle_model)
    torch_names = get_layer_name_torch(torch_model)
    names = list(set(paddle_names).intersection(set(torch_names)))
    # with open(os.path.join(save_path, 'model_diff.txt'), 'w') as f:
    #     diff_names = sorted(list(set(paddle_names).difference(set(torch_names))))
    #     f.write("两个模型的diff层，paddle有而torch没有：\n")
    #     f.write('\n'.join(diff_names))
        
    #     diff_names = sorted(list(set(torch_names).difference(set(paddle_names))))
    #     f.write("\n\n两个模型的diff层，torch有而paddle没有：\n")
    #     f.write('\n'.join(diff_names))

    #     f.write('\n\n两个模型共有的层:\n')
    #     f.write('\n'.join(sorted(names)))
    names = sorted(names)
    paddle_handles = _register_forward_post_hook_paddle(names, paddle_model)
    torch_handles = _register_forward_post_hook_torch(names, torch_model)
    with paddle.no_grad():
        paddle_out = paddle_model(paddle_inputs)
    # add_log(paddle_log, names)
    clean_handles(paddle_handles)
    _feats.clear()
    paddle_log.add('out', paddle_out.numpy())
    paddle_log.save(os.path.join(save_path, 'data/', 'paddle_out'))

    del paddle_log

    torch_log = ReprodLogger()
    

    if torch_features:
        with torch.no_grad():
            torch_out = torch_model.forward_features(torch_inputs)
    else:
        with torch.no_grad():
            torch_out = torch_model(torch_inputs)

    # add_log(torch_log, names)
    clean_handles(torch_handles)
    _feats.clear()
    torch_log.add('out', torch_out.detach().cpu().numpy())
    torch_log.save(os.path.join(save_path, 'data/', 'torch_out'))


def check(inputs, paddle_model, torch_model, save_path='./align', **kwargs):
    paddle_model.eval()
    torch_model.eval()
    torch_model = torch_model.cpu()

    forward(inputs, paddle_model, torch_model, save_path=save_path, **kwargs)

    diff = ReprodDiffHelper()
    paddle_info = diff.load_info(os.path.join(save_path, 'data/', 'paddle_out.npy'))
    torch_info = diff.load_info(os.path.join(save_path, 'data/', 'torch_out.npy'))
    diff.compare_info(paddle_info, torch_info)
    diff.report(path=os.path.join(save_path, 'diff.txt'))
