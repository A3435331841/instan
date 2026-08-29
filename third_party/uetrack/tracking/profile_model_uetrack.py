import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import numpy as np
from lib.test.evaluation.environment import env_settings
import os
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='uetrack',
                        help='training script name')
    parser.add_argument('--config', type=str, default='uetrack_base', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, text, template_list, search_list, template_anno_list, text_src, task_index, xz, bs):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # textencoder
    macs1, params1 = profile(model, inputs=(text, None, None, None, None, None, None, None, "text_dis"), custom_ops=custom_ops ,verbose=True)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('text encoder macs is ', macs)
    print('text encoder params is ', params)
    # encoder
    macs2, params2 = profile(model, inputs=(None, template_list, search_list, template_anno_list, text_src, task_index, None,None, "encoder"), custom_ops=custom_ops ,verbose=True)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('encoder macs is ', macs)
    print('encoder params is ', params)
    # tracking decoder
    macs3, params3 = profile(model, inputs=(None, None, None, None, None, None, xz, None, "tracking_decoder"), custom_ops=custom_ops, verbose=True)
    macs, params = clever_format([macs3, params3], "%.3f")
    print('tracking decoder macs is ', macs)
    print('tracking decoder params is ', params)
    # task_decoder
    macs4, params4 = profile(model, inputs=(None, None, None, None, None, None, xz, None,  "task_decoder"), custom_ops=custom_ops, verbose=True)
    macs, params = clever_format([macs4, params4], "%.3f")
    print('task decoder macs is ', macs)
    print('task decoder params is ', params)
    # the whole model
    macs, params = clever_format([macs2 + macs3, params2 + params3], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_encoder( template_list, search_list, template_anno_list, text_src, task_index)
            _ = model.forward_decoder(feature=xz)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
            _ = model.forward_decoder(feature=xz)
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 6, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:0"
    # device = "cpu"
    # torch.cuda.set_device(device)
    # Compute the Flops and Params of our model
    args = parse_args()
    '''update cfg'''
    prj_dir = env_settings().prj_dir
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))

    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    '''import network module'''
    model_module = importlib.import_module('lib.models.uetrack')
    model_constructor = model_module.build_uetrack_inference
    model = model_constructor(cfg)
    # get the template and search
    text = torch.zeros([1,77], dtype=torch.int64)
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    # transfer to device
    model = model.to(device)
    text = text.to(device)
    template = template.to(device)
    search = search.to(device)
    model.eval()
    # evaluate the model properties
    template_list = [template]
    search_list = [search]
    template_anno_list = [torch.tensor([[0.2117, 0.2794, 0.5766, 0.4411]], device=device, dtype=torch.float64)]
    task_index = torch.tensor([0], device=device, dtype=torch.int64)
    # text_src = model(text_data=text,mode='text')
    text_src = model.forward_textencoder_inference(text_data=text)
    xz,_,_ = model.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)

    evaluate(model, text, template_list, search_list, template_anno_list, text_src, task_index, xz, bs=bs)