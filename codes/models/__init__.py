import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srgan_r':
        from .SRGAN_R_model import SRGANModel as M
    elif model == 'srgan_al':
        from .SRGAN_al_model import SRGANModel as M
    elif model == 'srgan_an':
        from .SRGAN_an_model import SRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
