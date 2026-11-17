"""Bounded EPIGRIDS study. Original MATPOWER bytes are never modified.

Usage: python run.py --data <dataset> --output <new-dir> [--case NAME]
SPDX-License-Identifier: MIT
Data: Snodgrass et al., WISPO-POP/SyntheticElectricNetworkModels, CC BY 4.0.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, pathlib, re, time, traceback
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog
from matpowercaseframes import CaseFrames
from pypower.api import runopf, runpf, ppoption
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
import lp_ip_craft as lp
import sqp_craft as sqp


def clean(v):
    if dataclasses.is_dataclass(v): return clean(dataclasses.asdict(v))
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def save(p, v):
    pathlib.Path(p).write_text(json.dumps(clean(v),indent=2,allow_nan=False))


def load(path):
    """Constrained numeric literal parser; does not evaluate MATLAB."""
    raw=pathlib.Path(path).read_bytes(); text=raw.decode()
    text=re.sub(r'%[^\n]*','',text)
    d={'version':'2','baseMVA':float(re.search(r'mpc.baseMVA\s*=\s*([\d.eE+-]+)\s*;',text)[1])}
    allowed={'version','baseMVA','bus','gen','branch','gencost','bus_name'}
    unknown=set(re.findall(r'mpc\.(\w+)\s*=',text))-allowed
    if unknown: raise ValueError('unsupported arrays '+repr(unknown))
    for name in ['bus','gen','branch','gencost']:
        body=re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\];',text,re.S)[1]
        rows=[]
        for line in body.strip().splitlines():
            line=line.strip().rstrip(';')
            if not line: continue
            if not re.fullmatch(r'[\s\deE.+-]+',line): raise ValueError('not a numeric literal')
            rows.append([float(x) for x in line.split()])
        d[name]=np.array(rows)
    # Reference loader independently parses the ORIGINAL file.
    cf=CaseFrames(str(path)); diffs={}
    for name in ['bus','gen','branch','gencost']:
        r=getattr(cf,name).to_numpy(dtype=float)
        diffs[name]=float(np.max(np.abs(d[name]-r)))
        if not np.array_equal(d[name],r): raise AssertionError('loader mismatch '+name)
    if cf.baseMVA!=d['baseMVA']: raise AssertionError('baseMVA')
    ref={'version':'2','baseMVA':cf.baseMVA,**{k:getattr(cf,k).to_numpy(dtype=float) for k in diffs}}
    return d,ref,{'sha256':hashlib.sha256(raw).hexdigest(),'loader_max_diffs':diffs}


class Grid:
    def __init__(self,d):
        self.d=d; self.base=d['baseMVA']; self.bus=d['bus'].copy()
        self.gen=d['gen'][d['gen'][:,7]>0].copy(); self.br=d['branch'][d['branch'][:,10]>0].copy()
        ids={int(i):j for j,i in enumerate(self.bus[:,0])}
        self.f=np.array([ids[int(i)] for i in self.br[:,0]])
        self.t=np.array([ids[int(i)] for i in self.br[:,1]])
        self.gb=np.array([ids[int(i)] for i in self.gen[:,0]])
        self.nb=len(self.bus);self.ng=len(self.gen);self.nl=len(self.br)
        refs=np.flatnonzero(self.bus[:,1]==3)
        if len(refs)!=1: raise ValueError('expected one slack')
        self.ref=int(refs[0]);self.free=np.delete(np.arange(self.nb),self.ref)
        self.Cg=sp.csr_matrix((np.ones(self.ng),(self.gb,np.arange(self.ng))),shape=(self.nb,self.ng))
        self.Cf=sp.csr_matrix((np.ones(self.nl),(np.arange(self.nl),self.f)),shape=(self.nl,self.nb))
        self.Ct=sp.csr_matrix((np.ones(self.nl),(np.arange(self.nl),self.t)),shape=(self.nl,self.nb))
        tap=self.br[:,8].copy();tap[tap==0]=1
        tap=tap*np.exp(1j*np.deg2rad(self.br[:,9]));ys=1/(self.br[:,2]+1j*self.br[:,3])
        bc=1j*self.br[:,4]/2
        self.yff=(ys+bc)/abs(tap)**2; self.ytt=ys+bc
        self.yft=-ys/np.conj(tap); self.ytf=-ys/tap
        self.Yf=sp.diags(self.yff)@self.Cf+sp.diags(self.yft)@self.Ct
        self.Yt=sp.diags(self.ytf)@self.Cf+sp.diags(self.ytt)@self.Ct
        self.Y=self.Cf.T@self.Yf+self.Ct.T@self.Yt+sp.diags((self.bus[:,4]+1j*self.bus[:,5])/self.base)
        self.Y=self.Y.tocsr()
        components=sp.csgraph.connected_components(abs(self.Y),directed=False)[0]
        if components!=1: raise ValueError('unexplained islands')
        co=[]
        for row in d['gencost'][d['gen'][:,7]>0]:
            if row[0]!=2: raise ValueError('unsupported nonpolynomial cost')
            c=np.trim_zeros(row[4:4+int(row[3])],'f')
            if len(c)>3: raise ValueError('nonquadratic cost')
            c=np.pad(c,(3-len(c),0));co.append(c)
        self.co=np.array(co)
        if np.any(self.co[:,0]<0): raise ValueError('nonconvex cost')
        self.rated=np.flatnonzero(self.br[:,5]>0)
        # MATPOWER ignores a pair of zero angle bounds.
        self.ang=np.flatnonzero(((self.br[:,11]!=0)|(self.br[:,12]!=0))&((self.br[:,11]>-360)|(self.br[:,12]<360)))
        self.T=(self.Cf-self.Ct).tocsr()
        self.na=self.nb-1;self.n=self.na+self.nb+2*self.ng
        self.vm=slice(self.na,self.na+self.nb);self.pg=slice(self.vm.stop,self.vm.stop+self.ng);self.qg=slice(self.pg.stop,self.n)
        self.lo=np.r_[np.full(self.na,-np.inf),self.bus[:,12],self.gen[:,9]/self.base,self.gen[:,4]/self.base]
        self.hi=np.r_[np.full(self.na,np.inf),self.bus[:,11],self.gen[:,8]/self.base,self.gen[:,3]/self.base]
        self.cl=np.r_[np.zeros(2*self.nb),np.full(2*len(self.rated),-np.inf),np.deg2rad(self.br[self.ang,11])]
        self.cu=np.r_[np.zeros(2*self.nb),np.tile((self.br[self.rated,5]/self.base)**2,2),np.deg2rad(self.br[self.ang,12])]
        adj=self.Y.copy();adj.data=np.ones(len(adj.data))
        z=sp.csr_matrix((self.nb,self.ng));a=adj[:,self.free]
        flow=(self.Cf+self.Ct)[self.rated];fz=sp.csr_matrix((len(self.rated),self.ng))
        self.pattern=sp.vstack([sp.hstack([a,adj,self.Cg,z]),sp.hstack([a,adj,z,self.Cg]),
            sp.hstack([flow[:,self.free],flow,fz,fz]),sp.hstack([flow[:,self.free],flow,fz,fz]),
            sp.hstack([self.T[self.ang][:,self.free],sp.csr_matrix((len(self.ang),self.nb+2*self.ng))])]).tocsr()
        self.pattern.sort_indices();self.rows=np.repeat(np.arange(len(self.cl)),np.diff(self.pattern.indptr));self.cols=self.pattern.indices
        self.scale=10000.
        self.functions_calls=0;self.derivative_calls=0

    def x_from(self,d,clip=True):
        theta=np.deg2rad(d['bus'][:,8]);theta-=theta[self.ref]
        gen=d['gen'][d['gen'][:,7]>0]
        x=np.r_[theta[self.free],d['bus'][:,7],gen[:,1]/self.base,gen[:,2]/self.base]
        return np.clip(x,self.lo,self.hi) if clip else x

    def unpack(self,x):
        x=np.asarray(x);a=np.zeros(self.nb);a[self.free]=x[:self.na]
        return x,np.asarray(x[self.vm])*np.exp(1j*a),a

    def cost(self,pg):
        p=np.asarray(pg)*self.base
        return float(np.sum(self.co[:,0]*p*p+self.co[:,1]*p+self.co[:,2]))

    def functions(self,x):
        self.functions_calls+=1;x,v,a=self.unpack(x)
        s=v*np.conj(self.Y@v);mis=s+(self.bus[:,2]+1j*self.bus[:,3])/self.base-self.Cg@(x[self.pg]+1j*x[self.qg])
        sf=v[self.f]*np.conj(self.Yf@v);st=v[self.t]*np.conj(self.Yt@v)
        return self.cost(x[self.pg])/self.scale,np.r_[mis.real,mis.imag,abs(sf[self.rated])**2,abs(st[self.rated])**2,self.T[self.ang]@a]

    def jac(self,x):
        x,v,a=self.unpack(x);i=self.Y@v;e=np.exp(1j*a)
        dv=sp.diags(v);di=sp.diags(i)
        da=1j*dv@(di-self.Y@dv).conjugate()
        dm=dv@(self.Y@sp.diags(e)).conjugate()+sp.diags(np.conj(i)*e)
        z=sp.csr_matrix((self.nb,self.ng))
        blocks=[sp.hstack([da.real[:,self.free],dm.real,-self.Cg,z]),sp.hstack([da.imag[:,self.free],dm.imag,z,-self.Cg])]
        for C,Yb in [(self.Cf,self.Yf),(self.Ct,self.Yt)]:
            vi=C@v; ib=Yb@v; s=vi*np.conj(ib)
            dba=1j*(sp.diags(np.conj(ib))@C@dv-sp.diags(vi)@(Yb@dv).conjugate())
            dbm=sp.diags(np.conj(ib))@C@sp.diags(e)+sp.diags(vi)@(Yb@sp.diags(e)).conjugate()
            d2a=2*(sp.diags(np.conj(s))@dba).real;d2m=2*(sp.diags(np.conj(s))@dbm).real
            blocks.append(sp.hstack([d2a[self.rated][:,self.free],d2m[self.rated],sp.csr_matrix((len(self.rated),2*self.ng))]))
        blocks.append(sp.hstack([self.T[self.ang][:,self.free],sp.csr_matrix((len(self.ang),self.nb+2*self.ng))]))
        return sp.vstack(blocks).tocsr()

    def derivatives(self,x):
        self.derivative_calls+=1;x=np.asarray(x);g=np.zeros(self.n)
        g[self.pg]=(2*self.co[:,0]*x[self.pg]*self.base+self.co[:,1])*self.base/self.scale
        j=self.jac(x);return g,np.asarray(j[self.rows,self.cols]).ravel()

    def spec(self,hessian_pattern=None):
        groups=[[i,self.nb+i] for i in range(self.nb)]
        groups += [[2*self.nb+i,2*self.nb+len(self.rated)+i] for i in range(len(self.rated))]
        groups += [[2*self.nb+2*len(self.rated)+i] for i in range(len(self.ang))]
        return sqp.ProblemSpec(lower=self.lo,upper=self.hi,constraint_lower=self.cl,constraint_upper=self.cu,
            jacobian_indptr=self.pattern.indptr,jacobian_indices=self.pattern.indices,elements=groups,
            objective_pattern=list(range(self.pg.start,self.pg.stop)),linear_rows=list(range(2*self.nb+2*len(self.rated),len(self.cl))),
            **({} if hessian_pattern is None else {'hessian_indptr':hessian_pattern.indptr,'hessian_indices':hessian_pattern.indices}))


class ExactGrid(Grid):
    """Optional sparse analytic Lagrangian Hessian using PYPOWER derivatives.

    The AC equations/Jacobian and final physics checker retain their previous
    implementations. Directional finite differences check this extra callback.
    """
    def __init__(self,d):
        super().__init__(d)
        adj=self.Y.copy();adj.data=np.ones(adj.nnz)
        voltage=sp.bmat([[adj[self.free][:,self.free],adj[self.free]],
                         [adj[:,self.free],adj]],format='csr')
        self.hp=sp.tril(sp.block_diag([voltage,sp.eye(2*self.ng)],format='csr'),format='csr')
        self.hp.sort_indices()
        self.hr=np.repeat(np.arange(self.n),np.diff(self.hp.indptr));self.hc=self.hp.indices

    def spec(self):return super().spec(hessian_pattern=self.hp)

    def hessian_matrix(self,x,lam):
        from pypower.d2Sbus_dV2 import d2Sbus_dV2
        from pypower.d2ASbr_dV2 import d2ASbr_dV2
        from pypower.dSbr_dV import dSbr_dV
        x,v,a=self.unpack(x);lam=np.asarray(lam)
        p=d2Sbus_dV2(self.Y,v,lam[:self.nb]);q=d2Sbus_dV2(self.Y,v,lam[self.nb:2*self.nb])
        blocks=[-pp.real-qq.imag for pp,qq in zip(p,q)]
        branch=self.br.copy();branch[:,0]=self.f;branch[:,1]=self.t
        dfa,dfm,dta,dtm,sf,st=dSbr_dV(branch,self.Yf,self.Yt,v)
        for side,C,Y,da,dm,s in [(0,self.Cf,self.Yf,dfa,dfm,sf),(1,self.Ct,self.Yt,dta,dtm,st)]:
            mu=np.zeros(self.nl);start=2*self.nb+side*len(self.rated);mu[self.rated]=-lam[start:start+len(self.rated)]
            blocks=[b+z for b,z in zip(blocks,d2ASbr_dV2(da,dm,s,C,Y,v,mu))]
        aa,av,va,vv=blocks
        voltage=sp.bmat([[aa[self.free][:,self.free],av[self.free]],
                        [va[:,self.free],vv]],format='csr')
        cost=sp.diags(np.r_[2*self.co[:,0]*self.base**2/self.scale,np.zeros(self.ng)])
        return sp.block_diag([voltage,cost],format='csr')

    def lagrangian_hessian(self,x,lam):
        h=self.hessian_matrix(x,lam)
        return np.asarray(h[self.hr,self.hc]).ravel()


def checker(g,x,reference):
    """Independent PYPOWER Ybus plus direct original branch-current replay."""
    x,v,a=g.unpack(x);r=ext2int(reference.copy());y,yf,yt=makeYbus(r['baseMVA'],r['bus'],r['branch'])
    mismatch=v*np.conj(y@v)+(g.bus[:,2]+1j*g.bus[:,3])/g.base-g.Cg@(x[g.pg]+1j*x[g.qg])
    sf=v[g.f]*np.conj(yf@v);st=v[g.t]*np.conj(yt@v)
    violations=np.r_[g.lo-x,x-g.hi,abs(sf[g.rated])-g.br[g.rated,5]/g.base,abs(st[g.rated])-g.br[g.rated,5]/g.base]
    if len(g.ang):violations=np.r_[violations,np.deg2rad(g.br[g.ang,11])-g.T[g.ang]@a,g.T[g.ang]@a-np.deg2rad(g.br[g.ang,12])]
    return {'balance_pu':float(np.max(abs(mismatch))),'max_MW_Mvar':float(max(abs(mismatch.real).max(),abs(mismatch.imag).max())*g.base),
            'limit_violation':float(max(0.,np.max(violations))),'cost':g.cost(x[g.pg]),'loss_MW':float((sf+st).real.sum()*g.base)}


def dc(g,mode='secant'):
    b=1/g.br[:,3];tap=g.br[:,8].copy();tap[tap==0]=1;b/=tap
    Bf=sp.diags(b)@g.T;Bbus=g.T.T@Bf;pfshift=-b*np.deg2rad(g.br[:,9]);pshift=g.T.T@pfshift
    n=g.na+2*g.ng;rows=[sp.hstack([Bbus[:,g.free],-g.Cg,sp.csr_matrix((g.nb,g.ng))])]
    rhs=-(g.bus[:,2]+g.bus[:,4])/g.base-pshift
    lo=list(rhs);hi=list(rhs)
    rows.append(sp.hstack([Bf[g.rated][:,g.free],sp.csr_matrix((len(g.rated),2*g.ng))]));rate=g.br[g.rated,5]/g.base
    lo.extend(-rate-pfshift[g.rated]);hi.extend(rate-pfshift[g.rated])
    if len(g.ang):
        rows.append(sp.hstack([g.T[g.ang][:,g.free],sp.csr_matrix((len(g.ang),2*g.ng))]));lo.extend(np.deg2rad(g.br[g.ang,11]));hi.extend(np.deg2rad(g.br[g.ang,12]))
    rr=[];cc=[];vv=[];intercepts=[];error=0
    for j,(a,b,c) in enumerate(g.co):
        points=np.linspace(g.gen[j,9],g.gen[j,8],17);h=points[1]-points[0];error+=a*h*h/4
        if mode=='tangent':lines=[(2*a*p+b,c-a*p*p) for p in points]
        else:lines=[(a*(p+q)+b,c-a*p*q) for p,q in zip(points[:-1],points[1:])]
        for slope,offset in lines:
            k=len(intercepts);rr.extend([k,k]);cc.extend([g.na+j,g.na+g.ng+j]);vv.extend([-slope*g.base/g.scale,1]);intercepts.append(offset/g.scale)
    rows.append(sp.csr_matrix((vv,(rr,cc)),shape=(len(intercepts),n)));lo.extend(intercepts);hi.extend([np.inf]*len(intercepts))
    A=sp.vstack(rows).tocsr();A.sort_indices();cost=np.r_[np.zeros(g.na+g.ng),np.ones(g.ng)]
    lower=np.r_[np.full(g.na,-np.inf),g.gen[:,9]/g.base,np.full(g.ng,-np.inf)]
    upper=np.r_[np.full(g.na,np.inf),g.gen[:,8]/g.base,np.full(g.ng,np.inf)]
    return A,cost,lower,upper,np.array(lo),np.array(hi),error


def solve_dc(g):
    records=[];candidate=None
    for mode in ['secant','tangent']:
        A,c,lb,ub,rl,ru,err=dc(g,mode);eq=rl==ru;low=np.isfinite(rl)&~eq;high=np.isfinite(ru)&~eq
        t=time.monotonic();h=linprog(c,A_ub=sp.vstack([-A[low],A[high]]),b_ub=np.r_[-rl[low],ru[high]],A_eq=A[eq],b_eq=rl[eq],bounds=list(zip(lb,ub)),method='highs',options={'time_limit':60,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        rec={'mode':mode,'variables':len(c),'rows':len(rl),'nnz':A.nnz,'highs_status':h.message,'highs_seconds':time.monotonic()-t,'secant_error_bound_currency':err}
        m=lp.Model(objective=c,lower=lb,upper=ub,row_lower=rl,row_upper=ru,indptr=A.indptr,indices=A.indices,values=A.data)
        t=time.monotonic();r=lp.solve(m,backend='native',options=lp.Options(threads=1),backend_options=lp.NativeOptions(time_limit_seconds=60,component_threads=1))
        rec.update(native_status=r.status,native_seconds=time.monotonic()-t,native_result=clean(r))
        if r.candidate is not None:
            x=np.array(r.candidate);viol=float(max(0,np.max(np.r_[rl-A@x,A@x-ru,lb-x,x-ub])))
            obj=float(c@x);rel=abs(obj-h.fun)/max(1,abs(h.fun)) if h.success else None
            rec.update(objective_currency=obj*g.scale,true_quadratic_cost=g.cost(x[g.na:g.na+g.ng]),max_violation=viol,relative_objective_difference=rel,accepted=bool(h.success and r.status=='optimal' and viol<=1e-7 and rel<=1e-6))
            if mode=='secant':candidate=x
        else:rec['accepted']=False
        if h.success:rec['highs_objective_currency']=h.fun*g.scale
        records.append(rec)
    return records,candidate


def derivative_check(g):
    rng=np.random.default_rng(644);x=g.x_from(g.d);x[g.vm]=(g.lo[g.vm]+g.hi[g.vm])/2
    errors=[]
    for k in range(4):
        d=rng.normal(size=g.n);d/=max(1,np.max(abs(d)));h=1e-6
        _,cp=g.functions(x+h*d);_,cm=g.functions(x-h*d);jd=g.jac(x)@d
        errors.append(float(np.max(abs((cp-cm)/(2*h)-jd))/max(1,np.max(abs(jd)))))
    return {'directional_relative_errors':errors,'accepted':max(errors)<2e-7}


def run_case(path,out,merit='l1',hessian='partitioned',qp_tolerance=1e-10,initial_hessian_scale=1.0):
    d,ref,ident=load(path);g=Grid(d);save(out/'import.json',{**ident,'buses':g.nb,'generators':g.ng,'branches':g.nl,'variables':g.n,'jacobian_nnz':g.pattern.nnz,'load_MW':g.bus[:,2].sum(),'minimum_generation_MW':g.gen[:,9].sum()})
    save(out/'derivatives.json',derivative_check(g))
    dcr,dcx=solve_dc(g);save(out/'dc.json',dcr)
    t=time.monotonic()
    opts=ppoption(VERBOSE=0,OUT_ALL=0,OPF_VIOLATION=1e-8,PDIPM_FEASTOL=1e-9,PDIPM_GRADTOL=1e-9,PDIPM_COMPTOL=1e-9,PDIPM_COSTTOL=1e-10,PDIPM_MAX_IT=150)
    refsol=runopf(ref,opts);refcheck=checker(g,g.x_from(refsol,clip=False),ref)
    save(out/'reference.json',{'success':refsol['success'],'seconds':time.monotonic()-t,'objective':refsol['f'],'checker':refcheck,'bus':refsol['bus'],'gen':refsol['gen'],'raw':refsol.get('raw')})
    starts={'source':g.x_from(d),'reference_diagnostic':g.x_from(refsol)}
    if dcx is not None:
        # Predetermined replay: original generator voltage setpoints, PQ reactive
        # demands, PV generator Q solved without bus switching; slack absorbs losses.
        replay={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in ref.items()}
        active=replay['gen'][:,7]>0
        replay['gen'][active,1]=dcx[g.na:g.na+g.ng]*g.base
        replay['bus'][:,8]=np.rad2deg(np.r_[0.,np.zeros(g.nb-1)])
        pf,success=runpf(replay,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-10,ENFORCE_Q_LIMS=False))
        px=g.x_from(pf)
        # x_from clips bounds; preserve the actual replay vector for independent check.
        px[g.vm]=pf['bus'][:,7];px[g.pg]=pf['gen'][active,1]/g.base;px[g.qg]=pf['gen'][active,2]/g.base
        save(out/'dc-ac-replay.json',{'success':bool(success),'checker':checker(g,px,ref),'bus_type_switching':False,'slack_absorbs_losses':True,'x':px})
        starts['dc_replay']=np.clip(px,g.lo,g.hi)
    else:save(out/'dc-ac-replay.json',{'success':False,'reason':'no native secant candidate'})
    ac=[]
    for name,x0 in starts.items():
        t=time.monotonic()
        try:
            model=ExactGrid(d) if hessian=='exact' else Grid(d);spec=model.spec()
            settings=sqp.SqpOptions(hessian=hessian,initial_hessian_scale=initial_hessian_scale,qp_solver='clarabel',max_iterations=300,max_callback_calls=5000,time_limit_seconds=60,stationarity_tolerance=1e-10,complementarity_tolerance=1e-10,qp_tolerance=qp_tolerance,qp_reuse_workspace=True,record_history=True,merit=merit)
            r=sqp.solve(spec,model,x0,method='sqp',method_options=settings)
            rec={'start':name,'seconds':time.monotonic()-t,'settings':clean(settings),'result':clean(r)}
            if r.candidate is not None:
                x=np.array(r.candidate);check=checker(g,x,ref);rec['checker']=check
                if r.multipliers is not None:
                    f,cv=g.functions(x);grad,jv=g.derivatives(x)
                    ev=sqp.Evaluation(objective=f,constraints=cv.tolist(),gradient=grad.tolist(),jacobian_values=jv.tolist(),primal_violation=0.,feasible_candidate=False)
                    kkt=sqp.kkt_residuals(spec,x,ev,r.multipliers);rec['fresh_kkt']=clean(kkt)
                    rec['stationarity_currency']=kkt.stationarity*g.scale;rec['complementarity_currency']=kkt.complementarity*g.scale
                refok=refsol['success'] and refcheck['balance_pu']<=1e-7 and refcheck['limit_violation']<=1e-7
                rec['cost_relative_to_reference']=(check['cost']-refsol['f'])/max(1,abs(refsol['f']))
                rec['accepted']=bool(r.status=='converged' and check['balance_pu']<=1e-7 and check['limit_violation']<=1e-7 and rec.get('stationarity_currency',np.inf)<=1e-6 and rec.get('complementarity_currency',np.inf)<=1e-6 and r.dual_sign is not None and r.dual_sign<=1e-8 and refok and rec['cost_relative_to_reference']<=.001)
            else:rec['accepted']=False
        except Exception as exc:rec={'start':name,'accepted':False,'exception':repr(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-t}
        ac.append(rec);save(out/'ac.json',ac)
        print(path.stem,name,rec.get('result',{}).get('status'),rec.get('checker'),rec.get('stationarity_currency'),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--case',default='all');ap.add_argument('--merit',choices=['l1','augmented-lagrangian'],default='l1');ap.add_argument('--hessian',choices=['partitioned','exact'],default='partitioned');ap.add_argument('--qp-tolerance',type=float,default=1e-10);ap.add_argument('--initial-hessian-scale',type=float,default=1.0);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cases=['Easy/Winter12Generous.m','Easy/Fall116Generous.m','Easy/Winter68Generous.m','BaseCase.m','Hard/Spring13Tight.m']
    if a.case!='all':cases=[c for c in cases if pathlib.Path(c).stem==a.case]
    protocol={'schema':'research.power/1','cases':cases,'curvature':['partitioned'],'exact_hessian':'optional not implemented','starts':['source','reference_diagnostic','dc_replay_if_available'],'cost_scale':10000,'dc_intervals':16,'native_limits':{'seconds':60,'steps':300,'callbacks':5000},'acceptance':{'primal':1e-7,'stationarity_currency':1e-6,'complementarity_currency':1e-6,'objective_relative':.001},'threads':1,'protocol_sha256':None}
    protocol['merit']=a.merit
    protocol['curvature']=[a.hessian]
    protocol['qp_tolerance']=a.qp_tolerance
    protocol['initial_hessian_scale']=a.initial_hessian_scale
    protocol['exact_hessian']='optional analytic PYPOWER derivatives, finite-difference checked'
    protocol['executed_source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    protocol['protocol_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();save(a.output/'protocol.json',protocol)
    for case in cases:
        o=a.output/pathlib.Path(case).stem;o.mkdir()
        try:run_case(a.data/'New England/MatPower Cases'/case,o,merit=a.merit,hessian=a.hessian,qp_tolerance=a.qp_tolerance,initial_hessian_scale=a.initial_hessian_scale)
        except Exception as e:save(o/'failure.json',{'exception':repr(e),'traceback':traceback.format_exc()});print(case,repr(e),flush=True)


if __name__=='__main__':main()
