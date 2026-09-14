#!/usr/bin/env python3
"""Numerical checks of the displayed algebra; these do not replace proofs."""
from pathlib import Path
import numpy as np
import ast
import bisect
ROOT=Path(__file__).resolve().parents[1]
count=0

def check(condition, label):
    global count
    if not condition:
        raise AssertionError(label)
    count+=1
    print('PASS',label)

rng=np.random.default_rng(9152026)
# The explicit gradient of the ratio functional, allowing correlated totals.
for trial in range(12):
    c=4;x=rng.uniform(.5,8,4*c)
    def fun(x):
        s,r,v0,v1=x.reshape(4,c);b=s/r
        return np.sum(b*v1)/np.sum(b*v0)
    s,r,v0,v1=x.reshape(4,c);b=s/r;m0=np.sum(b*v0);theta=fun(x)
    grad=np.concatenate([(v1-theta*v0)/(r*m0),-s*(v1-theta*v0)/(r*r*m0),-theta*b/m0,b/m0])
    eps=1e-5
    num=np.array([(fun(x+np.eye(len(x))[j]*eps)-fun(x-np.eye(len(x))[j]*eps))/(2*eps) for j in range(len(x))])
    check(np.allclose(grad,num,atol=1e-8,rtol=1e-7),f'generic-carrier gradient {trial+1}')
# Discrete summation by parts for arbitrary, not necessarily monotone rewards.
for trial in range(12):
    p=rng.dirichlet(np.ones(8));q=rng.dirichlet(np.ones(8));h=rng.normal(size=8)
    lhs=np.dot(h,p-q);rhs=-np.dot(np.diff(h),np.cumsum(p-q)[:-1])
    check(np.isclose(lhs,rhs,atol=1e-12),f'ordered-grid identity {trial+1}')
# Cancellation identity and recorded/linked unlinked-share conversion.
for trial in range(12):
    nr0,nr1=rng.uniform(.1,3,2); r0,r1=rng.uniform(.05,1.6,2)
    delta_r=nr1/nr0;delta_m=r1*nr1/(r0*nr0)
    check(np.isclose(delta_r,(r0/r1)*delta_m),f'burden comparison identity {trial+1}')
    q0,q1=rng.uniform(0,.8,2)
    check(np.isclose((r1*nr1/(1-q1))/(r0*nr0/(1-q0)),delta_m*(1-q0)/(1-q1)),f'unlinked-share identity {trial+1}')
# The resampling design must not normalise all patient effects by a sample mean.
source=(ROOT/'simulation/simulate_linked_reward.py').read_text()
check('frail /= frail.mean()' not in source,'independent patient-effect normalisation')
check(source.count('frail /= (1.2 if scenario == "G" else 4.0)')==3,'same population normalisation in all three generators')
# Exercise the actual linkage function without importing the protected-data scan.
for name in ['scan_linkage.py','scan_linkage_features.py']:
    text=(ROOT/'scan'/name).read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='link')
    namespace={'bisect':bisect,'W':30}
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),namespace)
    link=namespace['link']
    att,unl=link([(0,100),(45,80)],[(60,70)])
    check(att==[[],[(60,70)]] and unl==0,f'30-second search restriction: {name}')
    att,unl=link([(0,100)],[(60,70)])
    check(att==[[]] and unl==12,f'overlap outside search remains unlinked: {name}')
    att,unl=link([(10,20)],[(20,25)])
    check(att==[[]] and unl==7,f'strictly positive overlap convention: {name}')
print(f'{count} additional algebra and design checks passed.')
