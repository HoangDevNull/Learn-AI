from cvxopt import matrix,solvers

# example  6
# c = matrix([-6.,-14.,-13])
# G = matrix([[0.5,1.,1.,0.,0.],[2.,2.,0.,-1.,0.],[1.,4.,0.,0.,-1.]])
# h = matrix([24.,60,0.,0.,0.])

#example 14
# c = matrix([-5., -3.])
# G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
# h = matrix([10., 16., 32., 0., 0.])

#example 16 
c = matrix([1., -2.,-4.,2.])
G = matrix([[1., 0.,-2.,-1.,0.,0.,0.],[0.,1.,1.,0.,-1.,0.,0.],[-2.,0.,8.,0.,0.,-1.,0.],[0.,-1.,1.,0.,0.,0.,-1]])
h = matrix([4., 8.,- 12., 0., 0.,0.,0.])

sol = solvers.lp(c,G,h)

print(sol['x'])
print(-sol['primal objective'])