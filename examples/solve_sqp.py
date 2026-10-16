"""Native nonlinear solve using the grouped protocol and a checked derivative model."""
from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve
class Circle:
    def functions(self,x): return .5*((x[0]-2)**2+x[1]**2),[x[0]**2+x[1]**2]
    def derivatives(self,x): return [x[0]-2,x[1]],[2*x[0],2*x[1]]
spec=ProblemSpec(lower=[-3.,-3.],upper=[3.,3.],constraint_lower=[-float("inf")],constraint_upper=[1.],jacobian_indptr=[0,2],jacobian_indices=[0,1])
assert check_derivatives(spec,Circle(),[.2,.4]).passed
result=solve(spec,Circle(),[.2,.4],method="sqp",method_options=SqpOptions(record_qp_limit=2))
print(result)
assert result.status=="converged"
