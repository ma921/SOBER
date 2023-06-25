import torch
from botorch.test_functions import Hartmann
from botorch.test_functions import Shekel
from botorch.utils.transforms import unnormalize

hart6 = Hartmann(dim=6, negate=True)
shekel = Shekel(negate=True)

def AckleyFunction(x):
    # Ackley function
    a = 20
    b = 0.2
    c = 2 * torch.pi
    
    x = torch.atleast_2d(x)
    first = -a * torch.exp(
        -b * x.pow(2).mean(axis=1).sqrt()
    )
    second = (c * x).cos().mean(axis=1).exp()
    return -1*(first - second + a + torch.tensor(1).exp())
    
def BraninFunction(x):
    x = torch.atleast_2d(x)
    return ((x.sin() + (3*x).cos()/2).square() / ((x/2).square()+0.3)).prod(axis=1)    

def optimisant(x, i):
    return 100*(x[:,i+1] - x[:,i].pow(2)).pow(2) + (x[:,i] - 1).pow(2)

def RosenbrockFunction(x):
    x = torch.atleast_2d(x)
    n_data, n_dims = x.size()
    return -1 * torch.cat([
        optimisant(x, i) for i in range(n_dims - 1)
    ]).reshape(n_data, n_dims-1).mean(axis=1)

def HartmannFunction(x):
    return hart6(unnormalize(x, hart6.bounds))

def ShekelFunction(x):
    return shekel(x)