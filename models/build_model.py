import torch

from .model import *


def build_model(cfg):

    if cfg.model == 'PCReg_KPURes18_MSF':
        model = PCReg_KPURes18_MSF(cfg)
    elif cfg.model == "no_model":
        print("Warning: no model is being loaded; rarely correct thing to do")
        model = torch.nn.Identity()
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr':
        model = PCReg_KPURes18_MSF_loftr(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr1':
        model = PCReg_KPURes18_MSF_loftr1(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr2':
        model = PCReg_KPURes18_MSF_loftr2(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr3':
        model = PCReg_KPURes18_MSF_loftr3(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr4':
        model = PCReg_KPURes18_MSF_loftr4(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr5':
        model = PCReg_KPURes18_MSF_loftr5(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr6':
        model = PCReg_KPURes18_MSF_loftr6(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr7':
        model = PCReg_KPURes18_MSF_loftr7(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr8':
        model = PCReg_KPURes18_MSF_loftr8(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr9':
        model = PCReg_KPURes18_MSF_loftr9(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_loftr10':
        model = PCReg_KPURes18_MSF_loftr10(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_mamba':
        model = PCReg_KPURes18_MSF_mamba(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_redc':
        model = PCReg_KPURes18_MSF_redc(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_gau0':
        model = PCReg_KPURes18_MSF_gau0(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_gau1':
        model = PCReg_KPURes18_MSF_gau1(cfg)
    elif cfg.model == 'PCReg_KPURes18_MSF_mca0':
        model = PCReg_KPURes18_MSF_mca0(cfg)    
    
    else:
        raise ValueError("Model {} is not recognized.".format(cfg.name))

    """
    We will release code for baselines soon. This has been delayed since some of the
    baselines have particular licenses that we want to make sure we're respecting.
    """

    return model