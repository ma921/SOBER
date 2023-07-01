from torch import tril,triu, diagonal

def make_condition(x,m,c):
    v = diagonal(c,dim1=-2,dim2=-1).unsqueeze(-1)
    c_v = c/v
    m_c = m.unsqueeze(-2)+(x-m).unsqueeze(-1)*c_v
    m_cond = remove_diagonal(m_c)

    v_columns = c_v.unsqueeze(-3)
    cun = c.unsqueeze(-1)
    c_3d = cun*v_columns
    c_c = c.unsqueeze(-2)-c_3d
    c_cond = swapcut(c_c) # d x d x d -> d x d-1 x d-1
    return m_cond, c_cond

def remove_diagonal(MU):
    return tril(MU,-1)[...,:-1]+triu(MU,1)[...,1:]

def swapcut(Gamma):
    G = remove_diagonal(Gamma)
    G = G.swapaxes(-1,-3)
    G = remove_diagonal(G)
    G = G.swapaxes(-3,-2)
    return G


if __name__ == "__main__":
    import torch
    d = 4
    m = torch.rand(d,requires_grad=True,dtype=torch.float64)
    x = torch.zeros_like(m)
    a = torch.rand(d,d,dtype=torch.float64)
    c = torch.matmul(a,a.t()).unsqueeze(0)

    m_c, c_c = make_condition(x,m,c)
    for i in range(c.shape[0]):
        print("i="+str(i))
        C = c[i,...]
        print("c_ci")
        print(c_c[i,...])
        print(C[1:3,1:3]-1/C[0,0]*C[1:3,0].unsqueeze(-1)*C[1:3,0])
        c01 = torch.tensor([C[0,1],C[2,1]])
        print(torch.tensor([[float(C[0,0]),float(C[0,2])],[float(C[2,0]),float(C[2,2])]])-1/C[1,1]*c01.unsqueeze(-1)*c01)
        print(C[0:2,0:2]-1/C[2,2]*C[0:2,2].unsqueeze(-1)*C[0:2,2])


        print(m[1:3]+1/C[0,0]*C[1:3,0]*(x[0]-m[0]))
        print(m_c)
        print(x[0:2].unsqueeze(-2).shape)
        print(m_c.shape)
        print(c_c.shape)