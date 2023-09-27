import torch

from monai.losses import DiceCELoss


def get_loss(conf, train=True):
    conf_base = conf.base_loss

    if 'weight' in conf_base.params.keys():
        conf_base.params['weight'] = torch.tensor(conf_base.params['weight'])
    if 'ce_weight' in conf_base.params.keys():
        conf_base.params['ce_weight'] = torch.tensor(conf_base.params['ce_weight'])

    lossfun = eval(conf_base.name)(**conf_base.params)

    if train:
        if len(conf.loss) > 0:
            lossfun = eval(conf.loss.name)(lossfun, **conf.loss.params)
        
    return lossfun
