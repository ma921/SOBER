import torch


def setting_parameters():
    """
    Return:
       - device: torch.device, cpu or cuda
       - dtype: torch.dtype, torch.float or torch.double
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.double
    return device, dtype