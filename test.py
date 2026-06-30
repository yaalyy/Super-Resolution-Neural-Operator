import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        if coord.dim() == 4:
            batch_size, height, width, _ = coord.shape
            coord = coord.view(batch_size, height * width, 2)
            out_shape = (height, width)
        else:
            out_shape = None

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :].unsqueeze(2), cell)
            ql = qr
            preds.append(pred.squeeze(-1))
        pred = torch.cat(preds, dim=2)
        if out_shape is not None:
            pred = pred.view(pred.shape[0], pred.shape[1], *out_shape)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False, mcell=False, clamp_output=True):
    model.eval()
    device = next(model.parameters()).device

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)

    if eval_type is None:
        metric_fn = utils.calc_psnr
        scale = scale_max
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type in ('pde-rel-l2', 'rel-l2'):
        scale = scale_max
        metric_fn = utils.calc_rel_l2
    elif eval_type in ('pde-mse', 'mse'):
        scale = scale_max
        metric_fn = utils.calc_mse
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        cnt+=1

        batch = utils.batch_to_device(batch, device)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        if mcell == False: c = 1
        else : c = max(scale/scale_max, 1)


        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            if clamp_output:
                pred.clamp_(0, 1)
            res = metric_fn(pred, batch['gt'])

        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/test_srno.yaml')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    num_workers = spec.get('num_workers', 8)
    pin_memory = torch.cuda.is_available()
    collate_fn = utils.numpy_collate if num_workers > 0 else None
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
        collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_spec = torch.load(args.model, map_location=device)['model']
    model = models.make(model_spec, load_sd=True).to(device)
    

    import time
    t1= time.time()
    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = int(args.scale_max),
        verbose=True,
        mcell=str(args.mcell).lower() in ('true', '1', 'yes'),
        clamp_output=config.get('clamp_output', True))
    t2 =time.time()
    print('result: {:.4f}'.format(res), utils.time_text(t2-t1))
    
