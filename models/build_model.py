import torch

from .model import *


def build_model(cfg):
    if cfg.model == "no_model":
        print("Warning: no model is being loaded; rarely correct thing to do")
        model = torch.nn.Identity()
    elif cfg.model == 'FFMT':
        model = FFMT(cfg)
    elif cfg.model == 'FFMT1':
        model = FFMT1(cfg)
    elif cfg.model == 'FFMT2':
        model = FFMT2(cfg)
    elif cfg.model == 'FFMT3':
        model = FFMT3(cfg)
    elif cfg.model == 'FFMT4':
        model = FFMT4(cfg)
    elif cfg.model == 'FFMT5':
        model = FFMT5(cfg)
    elif cfg.model == 'FFMT6':
        model = FFMT6(cfg)
    elif cfg.model == 'FFMT7':
        model = FFMT7(cfg)
    elif cfg.model == 'FFMT8':
        model = FFMT8(cfg)
    elif cfg.model == 'FFMT9':
        model = FFMT9(cfg)
    elif cfg.model == 'FFMT10':
        model = FFMT10(cfg)
    elif cfg.model == 'FFMT_mamba':
        model = FFMT_mamba(cfg)
    elif cfg.model == 'FFMT_redc':
        model = FFMT_redc(cfg)
    elif cfg.model == 'FFMT_gau0':
        model = FFMT_gau0(cfg)
    elif cfg.model == 'FFMT_gau1':
        model = FFMT_gau1(cfg)
    elif cfg.model == 'FFMT_mca0':
        model = FFMT_mca0(cfg)    
    
    else:
        raise ValueError("Model {} is not recognized.".format(cfg.name))

    """
    We will release code for baselines soon. This has been delayed since some of the
    baselines have particular licenses that we want to make sure we're respecting.
    """

    return model