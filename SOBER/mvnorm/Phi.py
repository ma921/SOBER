from numpy import zeros_like as np_zeros_like
from torch import tensor, diagonal, zeros, tril_indices, tril, triu, diag_embed, erfc, from_numpy
from torch.autograd import Function

from .integration import hyperrectangle_integration
from .conditioning import make_condition


def phi(z,v):
    return 0.39894228040143270286321808271/v.sqrt()*(-z**2/(2*v)).exp()
    #              ^ oneOverSqrt2pi     

def Phi1D(x,m,c):
    z = (x-m)/c.squeeze(-1).sqrt()
    return (erfc(-z*0.70710678118654746171500846685)/2).squeeze(-1)
    #                  ^ sqrt(2)/2

def phi2_sub(z,C): # compute pairs of bivariate densities and organize them in a matrix
    V = diagonal(C,dim1=-2,dim2=-1)
    a,c = V.unsqueeze(-1),V.unsqueeze(-2)
    det = a*c-C**2
    x1 = z.unsqueeze(-1)
    x2 = z.unsqueeze(-2)
    exponent = -0.5/det*(c*x1**2+a*x2**2-2*C*x1*x2)
    res = 0.15915494309189534560822210096/det.sqrt()*(exponent).exp()
    #           ^ oneOver2pi     
    return tril(res,-1) + triu(res,1)


def to_torch(x):
    if len(x.shape) == 0:
        return tensor(float(x))
    else:
        return from_numpy(x)


class PhiHighDim(Function):

    @staticmethod
    def forward(ctx, m, c):
        m_np = m.numpy()
        c_np = .5*(c+c.transpose(-1,-2)).numpy()
        ctx.save_for_backward(m,c)
        res_np = hyperrectangle_integration(m_np,c_np)
        return to_torch(res_np)

    @staticmethod
    def backward(ctx,grad_output):
        if grad_output is None:
            return None, None
        res_m = res_c = None
        need_m, need_c = ctx.needs_input_grad[0:2]
        if need_c or need_m:
            m,c = ctx.saved_tensors
            m_cond,c_cond = make_condition(0,m,c)
            v = diagonal(c,dim1=-2,dim2=-1)
            p = phi(m,v)
            P = Phi(m_cond,c_cond)
            grad_m = -P*p
            grad_output_u1 = grad_output.unsqueeze(-1)
            res_m = grad_output_u1*grad_m
            if need_c:
                d = c.shape[-1] # d==1 should never happen here
                if d==2:
                    P2 = 1
                else:
                    trilind = tril_indices(d,d-1,offset=-1)
                    m_cond2,c_cond2 = make_condition(0,m_cond,c_cond)
                    Q_l = Phi(m_cond2[...,trilind[0],trilind[1],:],c_cond2[...,trilind[0],trilind[1],:,:])
                    P2 = zeros(*Q_l.shape[:-1],d,d,dtype=Q_l.dtype)
                    P2[...,trilind[0],trilind[1]] = Q_l
                    P2[...,trilind[1],trilind[0]] = Q_l
                p2 = phi2_sub(m,c)
                hess = p2*P2
                D = -(m*grad_m+(hess*c).sum(-1))/v
                grad_c = .5*(hess + diag_embed(D))
                res_c = grad_output_u1.unsqueeze(-1)*grad_c
        return res_m, res_c

Phinception = PhiHighDim.apply

def Phi(m,c):
    d = c.shape[-1]
    if d == 1:
        return Phi1D(0,m,c)
    else:
        return Phinception(m,c)

if __name__ == "__main__":
    import torch                                                                                          
                                                                                                      
    def jacobian(y, x, create_graph=False):                                                               
        jac = []                                                                                          
        flat_y = (y*1).reshape(-1) # does someone know why it doesn't work without this one?
        grad_y = torch.zeros_like(flat_y)                                                                 
        for i in range(len(flat_y)):                                                                      
            grad_y[i] = 1.                                                                                
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
        return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                        
    def hessian(y, x):                                                                                    
        return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                        
    def f(x):                                                                                             
        return x**2 +x[0]*x[1]*1
                                                                                                        
    x = torch.ones(4, requires_grad=True)                                                                 

    d = 20

    batch_shape = []

    m = torch.rand( d,requires_grad=True)#,dtype=torch.float64)
    a = torch.rand(*batch_shape,d,d)#,dtype=torch.float64)
    c = torch.matmul(a,a.transpose(-1,-2))
    
  
    P = (Phi(m,c))
    print("P =",end=" ")
    print(P)
    import time
    t1 = time.time()
    print(.5*hessian(P,m))
    t2 = time.time()
    print(t2-t1)
    c.requires_grad = True
    P = (Phi(m,c))
    t1 = time.time()
    print(grad(P,[c])[0])
    t2 = time.time()
    print(t2-t1)
