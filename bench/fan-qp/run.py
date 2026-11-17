"""Fan eligibility, lossless mapping and generic-QP baseline.

FanCyqlone is an optional external LGPL comparator, never silently replaced.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def save(p,d):pathlib.Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))


def eligible(branches,*,later_shared=False,cross_coupling=False,cone=False):
    if not branches:raise ValueError('empty fan')
    if len(set((b['N'],b['nx'],b['nu']) for b in branches))!=1:raise ValueError('unequal dimensions')
    if later_shared or cross_coupling or cone:raise ValueError('unsupported fan coupling/cone')
    if any(min(b['Q'])<=0 or min(b['QN'])<=0 or b['R']<=0 for b in branches):raise ValueError('nonpositive curvature')
    for b in branches:
        if (b['nx'],b['nu'])!=(2,1) or b['N']<2:raise ValueError('exporter supports two states, one input, at least two stages')
        shapes={'A':(2,2),'B':(2,1),'Q':(2,),'QN':(2,),'x0':(2,),'target':(2,)}
        if any(np.asarray(b[k]).shape!=shape or not np.all(np.isfinite(b[k])) for k,shape in shapes.items()):raise ValueError('invalid/nonfinite branch arrays')
        if not np.isfinite(b['R']) or not np.isfinite(b['weight']) or b['weight']<=0:raise ValueError('invalid curvature or branch weight')
    if abs(sum(b['weight'] for b in branches)-1)>1e-12:raise ValueError('branch weights must sum to one')
    return True


def fixture(J,x0):
    return [{'N':8,'nx':2,'nu':1,'A':[[1.,.1],[0.,.98]],'B':(s*np.array([[.005],[.1]])).tolist(),'x0':list(x0),'Q':[3.,.4],'QN':[6.,.8],'R':.2,'weight':1/J,'target':[1.,0.]} for s in ([.9] if J==1 else [.9,1.1])]


def assemble(branches,constrained=True):
    eligible(branches);J=len(branches);N=branches[0]['N'];names=[('u',0,0,0)];xi={};ui={(j,0):0 for j in range(J)}
    for j in range(J):
        for k in range(1,N+1):
            for s in range(2):xi[j,k,s]=len(names);names.append(('x',j,k,s))
        for k in range(1,N):ui[j,k]=len(names);names.append(('u',j,k,0))
    n=len(names);H=np.zeros(n);g=np.zeros(n);lo=np.full(n,-np.inf);hi=-lo;A=np.zeros((J*N*2,n));b=np.zeros(J*N*2);const=0
    for j,br in enumerate(branches):
        w=br['weight'];q=np.array(br['Q']);qn=np.array(br['QN']);target=np.array(br['target']);initial=np.array(br['x0'])
        for k in range(N):
            u=ui[j,k];H[u]+=w*br['R']
            if constrained:lo[u]=-2;hi[u]=2
            if k==0:const+=.5*w*np.sum(q*(initial-target)**2)
            else:
                for s in range(2):H[xi[j,k,s]]+=w*q[s];g[xi[j,k,s]]-=w*q[s]*target[s]
                const+=.5*w*np.sum(q*target**2)
            aa=np.array(br['A']);bb=np.array(br['B'])[:,0]
            for s in range(2):
                r=(j*N+k)*2+s;A[r,xi[j,k+1,s]]=1;A[r,u]=-bb[s]
                if k==0:b[r]=aa[s]@initial
                else:
                    for t in range(2):A[r,xi[j,k,t]]-=aa[s,t]
        for s in range(2):H[xi[j,N,s]]+=w*qn[s];g[xi[j,N,s]]-=w*qn[s]*target[s]
        const+=.5*w*np.sum(qn*target**2)
        if constrained:
            for k in range(1,N+1):lo[xi[j,k,0]]=-2;hi[xi[j,k,0]]=2
    assert n==J*N*3-(J-1)
    return H,g,A,b,lo,hi,const,names,xi,ui


def unpack(q,x,j,N):
    *_,xi,ui=q
    xs=np.array([[x[xi[j,k,s]] for s in range(2)] for k in range(1,N+1)])
    us=np.array([x[ui[j,k]] for k in range(N)])
    return xs,us


def direct(branches,q,x):
    obj=0.;res=[];rollout=0
    for j,br in enumerate(branches):
        N=br['N'];xs,us=unpack(q,x,j,N);states=np.vstack([br['x0'],xs]);a=np.array(br['A']);b=np.array(br['B'])[:,0];target=np.array(br['target']);Q=np.array(br['Q']);QN=np.array(br['QN']);w=br['weight']
        for k in range(N):
            obj+=.5*w*(np.sum(Q*(states[k]-target)**2)+br['R']*us[k]**2)
            res.extend(states[k+1]-a@states[k]-b*us[k])
        obj+=.5*w*np.sum(QN*(states[-1]-target)**2)
        state=np.array(br['x0'])
        for k in range(N):state=a@state+b*us[k];rollout=max(rollout,float(np.max(abs(state-states[k+1]))))
    return obj,np.array(res),rollout


def fan_arrays(branches):
    out=[]
    for b in branches:
        N=b['N'];AB=np.tile(np.column_stack([b['A'],b['B']]),(N,1,1));CD=np.zeros((N,2,3));CD[:,0,2]=1;CD[:,1,0]=1
        QRS=np.tile(np.diag([*b['Q'],b['R']])*b['weight'],(N,1,1));QN=np.diag(b['QN'])*b['weight'];qr=np.zeros(N*3+2)
        for k in range(N):qr[k*3:k*3+2]=-np.array(b['Q'])*b['target']*b['weight']
        qr[-2:]=-np.array(b['QN'])*b['target']*b['weight']
        rhs=np.zeros((N+1)*2);rhs[:2]=-np.array(b['x0'])
        out.append({'AB':AB.tolist(),'CD':CD.tolist(),'CN':np.array([[1.,0.]]).tolist(),'QRS':QRS.tolist(),'QN':QN.tolist(),'qr':qr.tolist(),'rhs_eq':rhs.tolist(),'rhs_lb':([-2.,-2.]*N)+[-2.],'rhs_ub':([2.,2.]*N)+[2.],'numpy_memory_order':'F required when constructing OCP','shared_first_input':'FanOCP owns nonanticipativity'})
    return out


def solve_native(binary,q,path):
    H,g,A,b,lo,hi,*_=q;nums=np.r_[len(H),len(b),H,g,lo,hi,b,b,A.ravel()];text=' '.join(format(float(x),'.17g') for x in nums)
    path.write_text(text);t=time.monotonic();r=subprocess.run([str(binary)],input=text,text=True,capture_output=True,timeout=30)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d['process_seconds']=time.monotonic()-t;return d


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol={'schema':'research.fanqp/1','branches':[1,2],'horizon':8,'x0':[[.4,-.1],[.2,-.15],[-.3,.2]],'constrained':[False,True],'thread_count':1,'stationarity':1e-8,'primal':1e-8,'complementarity':1e-6,'objective_relative':1e-7,'external_comparator':'not executed: Conan dependency build exceeded 1000-second ceiling','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'bridge_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest()}
    save(a.output/'protocol.json',protocol);results=[]
    for J in [1,2]:
        for start,x0 in enumerate(protocol['x0']):
            branches=fixture(J,x0);save(a.output/f'fan-{J}-{start}.json',fan_arrays(branches))
            for constrained in [False,True]:
                name=f'J{J}-s{start}-bounded{constrained}';q=assemble(branches,constrained);H,g,A,b,lo,hi,const,names,*_=q
                rng=np.random.default_rng(118);v=rng.normal(size=len(H));f,res,_=direct(branches,q,v)
                assert abs(f-(.5*np.dot(H*v,v)+g@v+const))<1e-12
                assert np.max(abs(res-(A@v-b)))<1e-14
                r=solve_native(a.binary,q,a.output/(name+'.txt'));x=np.array(r['x']);lam=np.array(r['lambda']);z=np.array(r['z']);obj,res,replay=direct(branches,q,x)
                stat=float(np.max(abs(H*x+g-A.T@lam-z)));primal=float(max(0,np.max(abs(res)),np.max(lo-x),np.max(x-hi)))
                comp=max([abs(z[i]*(x[i]-lo[i] if z[i]>0 else hi[i]-x[i])) for i in range(len(x)) if z[i]!=0 and np.isfinite(lo[i] if z[i]>0 else hi[i])]+[0.])
                if not constrained:
                    K=np.block([[np.diag(H),A.T],[A,np.zeros((len(b),len(b)))] ]);rhs=np.r_[-g,b];sol=np.linalg.solve(K,rhs);refx=sol[:len(H)]
                    backward=float(np.max(abs(K@sol-rhs))/(np.linalg.norm(K,np.inf)*np.linalg.norm(sol,np.inf)+np.linalg.norm(rhs,np.inf)));condition=float(np.linalg.cond(K))
                    reference={'method':'independent dense equality KKT / NumPy LAPACK','backward_residual':backward,'condition_2norm':condition,'success':True}
                else:
                    rr=minimize(lambda v:.5*np.dot(H*v,v)+g@v,np.zeros(len(H)),jac=lambda v:H*v+g,bounds=Bounds(lo,hi),constraints=LinearConstraint(A,b,b),method='SLSQP',options={'ftol':1e-12,'maxiter':400});refx=rr.x;reference={'method':'SciPy SLSQP','success':bool(rr.success),'message':rr.message}
                refobj=direct(branches,q,refx)[0];diff=abs(obj-refobj)/max(1,abs(refobj))
                accepted=bool(reference['success'] and primal<=1e-8 and stat<=1e-8 and comp<=1e-6 and diff<=1e-7 and replay<=1e-8)
                rec={'name':name,'J':J,'N':8,'x0':x0,'constrained':constrained,'n':len(H),'equalities':len(b),'hessian_nnz':int(np.count_nonzero(H)),'jacobian_nnz':int(np.count_nonzero(A)),'native':r,'objective_with_constant':obj,'reference':reference,'reference_objective':refobj,'relative_objective_difference':diff,'primal':primal,'stationarity':stat,'complementarity':comp,'rollout_error':replay,'accepted':accepted,'inverse_variable_order':names}
                results.append(rec);save(a.output/'results.json',results);print(name,accepted,obj,primal,stat,flush=True)
    # Explicit eligibility refusals; no approximate fan conversion.
    checks=[]
    cases=[([],{}),([dict(fixture(1,[.4,-.1])[0],R=-1)],{}),(fixture(2,[.4,-.1]),{'later_shared':True}),(fixture(2,[.4,-.1]),{'cross_coupling':True}),(fixture(2,[.4,-.1]),{'cone':True}),([fixture(1,[.4,-.1])[0],dict(fixture(1,[.4,-.1])[0],N=9)],{})]
    for br,kw in cases:
        try:eligible(br,**kw);checks.append(False)
        except ValueError:checks.append(True)
    assert all(checks);save(a.output/'eligibility-tests.json',{'rejected_unsupported':checks})


if __name__=='__main__':main()
