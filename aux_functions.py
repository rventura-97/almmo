from numpy import zeros, zeros_like, power, invert, diag, exp, vstack, \
    eye, arange, tile, repeat, sum, newaxis, transpose, einsum, sqrt, cos, pi
from numpy.linalg import norm
from sklearn.metrics import recall_score
import numpy as np

#CONSTANTS:
r0 = sqrt(2*(1-cos(pi/12))) # /12 # Initial cloud radius for normalized samples
eta0 = 0.1 # Forgetting factor 0.1
lambda0 = 0.8 # New rule density threshold
omega0 = 10 # Covariance matrices multiplier 10
epson0 = 0.4 # Feature removing threshold:

    
# AUXILIARY FUNCIONS:

def computeDistance(x1,x2):
    return norm(x1-x2,axis=0)

def normSample(x,mode='unitNorm'):
    if mode == 'unitNorm':
        xNorm = norm(x)
        if xNorm != 0:
            x = x/xNorm
        else:
            x = zeros_like(x)
    return x

def computeDensity(x,mu,X):
    D = zeros(x.shape[1])
    dx = power(norm(x-mu,axis=0),2)
    dX = X-power(norm(mu,axis=0),2)
    den = dx+dX
    idxs = (den==0)
    D[idxs]=1.0
    idxs = invert(idxs)
    D[idxs] = dX/den[idxs]
    return D

def computeAdjustedDensity(x,mu,X,M):
    D = zeros(M.shape[0])
    M_temp = M + 1
    mu_temp = mu*diag((M_temp-1)/M_temp) + x*(1/M_temp)
    X_temp = X*((M_temp-1)/M_temp)+ power(norm(x),2)*(1/M_temp)
    dx = power(norm(x-mu_temp,axis=0),2)
    dX = X_temp-power(norm(mu_temp,axis=0),2)
    den = dx+dX
    idxs = (den==0)
    D[idxs]=1.0
    idxs = invert(idxs)
    D[idxs] = dX[idxs]/den[idxs]
    return D

def computeLambda(dist,rho=1):
     return exp(-0.5*rho*power(dist,2))

def computeConseqExpMatrix(m,p):
    if p == 1:
        expMtx = expMtx = vstack((zeros(m), eye(m)))
        dim = expMtx.shape[0]
        return expMtx, dim
    else:
        expMtx = zeros(((p+1)**m,m))
        exps = arange(p+1)
        for i in range(m):
            if i == 0:
                expMtx[:,i] = tile(exps, (p+1)**(m-1))
            elif i == (m-1):
                expMtx[:,i] = repeat(exps,(p+1)**i)
            else:
                expMtx[:,i] = tile(repeat(exps,(p+1)**i),(p+1)**(m-1-i))
        expMtx = expMtx[(sum(expMtx, axis=1) <= p),:]
        dim = expMtx.shape[0]
        return expMtx, dim

def computeWeightedRLS(C,A,u,w,y,mode='tensor'):
    if mode == 'tensor':
        y=repeat(y[:,:,newaxis],w.size,axis=2)
        uT = transpose(u)
        Cu = einsum('ijk,jl->ilk',C,u)
        C -= einsum('i,jki->jki',w,einsum('ijk,jlk->ilk',einsum('ijk,jl->ilk',Cu,uT),C))/(1+einsum('i,jki->jki',w,einsum('ij,jkl->ikl',uT,Cu)))
        A += einsum('ijk,jlk->ilk',einsum('i,jki->jki',w,einsum('ijk,jl->ilk',C,u)),y-einsum('ij,jkl->ikl',uT,A))
    elif mode == 'matrix':
        uT = transpose(u)
        for i in range(w.size):
            Cu = C[:,:,i]*u
            C[:,:,i] -= (w[i]*Cu*uT*C[:,:,i])/(1+w[i]*uT*Cu)
            A[:,:,i] += w[i]*C[:,:,i]*u*(y-uT*A[:,:,i])
    return C, A

def specificity(y_true, y_pred):
    TN = np.sum(np.logical_and(np.where(y_pred==0,1,0),np.where(y_true==0,1,0)),dtype=np.float64)
    FP = np.sum(np.logical_and(np.where(y_pred==1,1,0),np.where(y_true==0,1,0)),dtype=np.float64)
    if TN + FP > 0:
        return TN/(TN + FP)
    else:
        return 0 
        
def geometric_mean(y_true, y_pred):
    return sqrt(recall_score(y_true, y_pred) * specificity(y_true, y_pred))
    