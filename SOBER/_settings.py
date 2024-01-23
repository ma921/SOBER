import torch

_device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu')
)
_dtype = torch.double


def setting_parameters(device=None, dtype=None):
    """
    Return:
       - device: torch.device, cpu or cuda
       - dtype: torch.dtype, torch.float or torch.double
    """
    global _device, _dtype
    if device:
        _device = device
    if dtype:
        _dtype = dtype
    return _device, _dtype
