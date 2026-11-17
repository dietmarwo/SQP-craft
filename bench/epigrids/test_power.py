"""Physics and parser boundary checks, independent of large-grid convergence."""
import importlib.util,pathlib,tempfile,unittest
import numpy as np

spec=importlib.util.spec_from_file_location('power_study',pathlib.Path(__file__).with_name('run.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def tiny():
    bus=np.zeros((2,13));bus[:,0]=[1,2];bus[:,1]=[3,1];bus[:,7]=1;bus[:,11]=1.1;bus[:,12]=.9
    gen=np.zeros((1,21));gen[0,[0,3,4,5,6,7,8,9]]=[1,100,-100,1,100,1,200,0]
    branch=np.zeros((1,13));branch[0,[0,1,3,5,10]]=[1,2,.1,200,1]
    return {'version':'2','baseMVA':100.,'bus':bus,'gen':gen,'branch':branch,'gencost':np.array([[2,0,0,3,.01,1,0.]])}


class Power(unittest.TestCase):
    def test_two_bus_hand_physics(self):
        g=p.Grid(tiny());x=g.x_from(g.d);x[0]=-.1
        v=np.array([1.,np.exp(-.1j)]);s=v*np.conj(g.Y@v)
        expected=np.sin(.1)/.1+1j*(1-np.cos(.1))/.1
        self.assertAlmostEqual(abs(s[0]-expected),0.,places=12)
        self.assertAlmostEqual(s.real.sum(),0.,places=12)
        self.assertTrue(p.derivative_check(g)['accepted'])
        result=p.checker(g,x,g.d)
        self.assertAlmostEqual(result['balance_pu'],abs(expected),places=12)

    def test_secant_envelope_error_bound(self):
        a=.01;b=2.;c=3.;points=np.linspace(0,200,17)
        for low,high in zip(points[:-1],points[1:]):
            x=(low+high)/2;sec=(a*(low+high)+b)*x+c-a*low*high
            self.assertAlmostEqual(sec-(a*x*x+b*x+c),a*(high-low)**2/4,places=11)

    def test_exact_hessian_matches_gradient_differences(self):
        g=p.ExactGrid(tiny());x=g.x_from(g.d);x[0]=-.1
        rng=np.random.default_rng(1701);lam=rng.uniform(-1,1,len(g.cl))
        h=g.hessian_matrix(x,lam);self.assertLess(abs(h-h.T).max(),1e-10)
        stored=np.asarray(h[g.hr,g.hc]).ravel()
        self.assertTrue(np.array_equal(stored,g.lagrangian_hessian(x,lam)))
        for direction in np.eye(g.n):
            step=1e-5
            def gradient(x):return g.derivatives(x)[0]-g.jac(x).T@lam
            observed=(gradient(x+step*direction)-gradient(x-step*direction))/(2*step)
            np.testing.assert_allclose(observed,h@direction,atol=1e-7,rtol=1e-6)

    def test_nonconvex_cost_is_refused(self):
        d=tiny();d['gencost'][0,4]=-.01
        with self.assertRaisesRegex(ValueError,'nonconvex'):p.Grid(d)

    def test_matlab_execution_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'bad.m';path.write_text("mpc.version='2';mpc.baseMVA=100;mpc.bus=[system('x')];")
            with self.assertRaises(ValueError):p.load(path)

    def test_unknown_array_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'bad.m';path.write_text("mpc.baseMVA=100;mpc.unknown=[1];")
            with self.assertRaisesRegex(ValueError,'unsupported'):p.load(path)


if __name__=='__main__':unittest.main()
