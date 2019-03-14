#Batch Normalization Forward 
def batch_norm(x, gamma, beta, eps):

  N, D = x.shape


  #모든 column 에 대한 평균값
  mu = 1./N * np.sum(x, axis = 0)

 
  #모든 element에서 mu를 뺀 N*D mtx
  xmu = x - mu
  sq = xmu ** 2

  #분산 (per-dimension variance) 
  var = 1./N * np.sum(sq, axis = 0)

  #엡실론을 더한 adjusted_sd 
  sqrtvar = np.sqrt(var + eps)

  #Inverse of adjusted_sd
  ivar = 1./sqrtvar

  #Normalization
  xhat = xmu * ivar

  #Scaling
  gammax = gamma * xhat
  #Shifting
  out = gammax + beta

  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache
#out: Batch Normalization
#cache: backprop. 을 위한 패러미터 값 반환  





#Backprop. of batch normalization (beta 와 gamma 학습) 
def batch_norm_backward(dout, cache):

  
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache


  N,D = dout.shape

  #Summation Gate 이기 때문에 1*sum(dout) 
  dbeta = np.sum(dout, axis=0)
  dgammax = dout 


  #Multi. Gate(xhat * gamma); 다른 변수가 남는다. 
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma


  #Multi. Gate(xmu * ivar); 다른 변수가 남는다. 
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar


  #Inverting Gate (1/sqrtvar)
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #Root squared (sqrt(var+eps)) 
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #Summation Gate 
  dsq = 1. /N * np.ones((N,D)) * dvar

  #Sqaured 
  dxmu2 = 2 * xmu * dsq

  #Substraction Gate 
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  dx2 = 1. /N * np.ones((N,D)) * dmu
  #Add two gradients of single node 
  dx = dx1 + dx2

  return dx, dgamma, dbeta
#dx : Derivative of loss function wrt Batch Normalized Layer 
#dgamma : Derivative of gamma wrt '' 
#dbeta : Derivative of beta wrt '' 