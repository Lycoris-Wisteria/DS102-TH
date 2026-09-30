import numpy as np
import copy
# x^4 + y^2 +2xy +1
#x^4 +2xy +3y^2 -4x +6y +10
def grad(u):
    x, y = u
    return np.array([
        4*x**3 +2*y -4,
        6*y + 2*x +6
    ])

def hessian(u):
    x, y = u
    return np.array([
        [12*x**2, 2],
        [2, 6]
    ])

def newton_optimizer(grad_fn, hessian_fn, u_0, tol=1e-8, max_iter=100):
    history = [u_0]
    u = copy.deepcopy(u_0)
    for _ in range(max_iter):
        grad = grad_fn(u)

        if np.linalg.norm(grad) <= tol:
            break 

        hessian = hessian_fn(u)
        #hessian_inv = np.linalg.inv(hessian)
        #u = grad @ hessian_inv
        #u = u - gra/hessian
        a = np.linalg.solve(hessian, grad)
        u = u - a
        history.append(copy.deepcopy(u))

    return history, u

u_0 = (1,1)
history, u = newton_optimizer(grad, hessian, u_0, max_iter=1_000)

for u in history:
    print(u)


