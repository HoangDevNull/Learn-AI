from cvxopt import matrix,solvers


# Example 1 
# Q = 2*matrix([ [2, .5], [.5, 1] ])
# p = matrix([1.0, 1.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 1.0], (1,2))
# b = matrix(1.0)
# sol=solvers.qp(Q, p, G, h, A, b)

# Example 11 
# Q = matrix([-10,-10])
# P = matrix([1,0],[0,1])
# G = matrix([1,2,1,-1,0],[1,1,4,0,-1])
# h = matrix([10., 16., 32., 0., 0])
# sol=solvers.qp(Q, p, G, h)
print(sol['x'])
print(-sol['primal objective'])