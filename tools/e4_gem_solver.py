"""E4 numeric repair: exact same regularized GEM dual QP, bounded least squares.

For Q=L L^T, min .5 v^T Q v + p^T v, v>=margin, is equivalent
(up to a constant) to min .5 ||L^T v - L^-1(-p)||^2 with same bounds.
No change to memory, gradients, margin, epsilon, or the native projection oracle.
"""
import numpy as np
import torch
from scipy.optimize import lsq_linear
SOLVER_VERSION='gem-same-qp-bvls-v1'
COUNTERS={'calls':0,'max_scaled_kkt_residual':0.0,'max_iterations':0}

def solve(matrix,linear,lower_bound):
 q=matrix.detach().cpu().double().numpy();p=linear.detach().cpu().double().numpy();chol=np.linalg.cholesky(q)
 fit=lsq_linear(chol.T,np.linalg.solve(chol,-p),bounds=(float(lower_bound),np.inf),method='bvls',tol=1e-12,max_iter=10000)
 v=fit.x;gradient=q@v+p;scale=max(1.,float(np.max(np.abs(p))),float(np.max(np.abs(q@v))))
 active=v<=float(lower_bound)+1e-8
 residual=max(float(np.max(np.abs(gradient[~active]))) if (~active).any() else 0.,float(np.max(np.maximum(-gradient[active],0.))) if active.any() else 0.)/scale
 assert fit.success and np.isfinite(v).all() and (v>=float(lower_bound)-1e-10).all() and residual<1e-7,('GEM BVLS KKT failure',fit.message,residual)
 COUNTERS['calls']+=1;COUNTERS['max_scaled_kkt_residual']=max(COUNTERS['max_scaled_kkt_residual'],residual);COUNTERS['max_iterations']=max(COUNTERS['max_iterations'],int(fit.nit))
 return torch.from_numpy(v.copy()).to(matrix.device,matrix.dtype)

def install():
 import gecko.algorithms.continual.gem as module
 module._solve_gem_dual_qp=solve
