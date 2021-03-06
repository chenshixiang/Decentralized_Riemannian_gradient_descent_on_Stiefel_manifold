from scipy import sparse
import numpy as np


class Quadratic:
    """"
     Quadratic function:  f_i(x) =  -  0.5* sum_{j=1}^num_instances (x'*a_i'*a_i*x)
       #:param num_instances: number of instances at local node
               dimension: dimension of variable
               m:  maximum number of instances across all nodes, only needed for decentralized alg
               data: matrix with size (num_instances, dimension)
               idx: row index of A and y
               eigengap: eigengap of matrix A: lambda_i/ lambda_{i+1}= eigengap
               sparse_mat: whether A is a csr sparse matrix
               x_true: the optimal solution
       #:return a sub-gradient vector:  partial f(x)
       #:return obj val: f(x)
      """
    def __init__(self, num_instances=0, dimension=0, col=0, A=np.empty([1,1]), m=1, eigengap=1, sparse_mat=False):
        self.num_instances = num_instances
        self.dimension = dimension
        self.col = col
        self.A = A
        self.m = m
        self.sparse_mat = sparse_mat
        self.eigengap = eigengap
        self.x_true = None

    def synthetic_data(self):
        """ generate synthetic data """
        self.A = np.random.randn(self.num_instances, self.dimension)
        u, singularvalue, vh = np.linalg.svd(self.A, full_matrices=False)
        top_s = singularvalue[0]
        self.x_true = vh
        singularvalue = np.array([top_s*self.eigengap**(i/2) for i in range(self.dimension)])
        self.A = np.matmul(u, vh * singularvalue[:, None])

    def obj(self, x, idx, n=1, return_objval=True):
        """"
            :param return_objval: whether return objective value
            :param sparse_mat: whether A is a csr sparse matrix
            :return gradient: A'*A*x
            :return obj val: 0.5*Tr(x'*A'*A*x)
        """
        if not self.sparse_mat:
            full_idx = np.arange(self.num_instances)
            if idx.size == full_idx.size and (idx == full_idx).all():
                """full grad """
                ax = np.matmul(self.A, x)
                grad = np.matmul(self.A.T, ax)
            elif idx.size == 0:
                """ empty idx """
                return np.zeros(x.shape), None
            else:
                """ batch grad """
                ax = np.matmul(self.A[idx, :], x)  # m*1
                grad = np.matmul(self.A[idx, :].T, ax)
            if return_objval is False:
                return -grad, None
            else:
                return -grad, -0.5 * np.sum(np.trace(np.matmul(ax.T, ax)))
        else:
            " sparse matrix "
            full_idx = np.arange(self.num_instances)
            if idx.size == full_idx.size and (idx == full_idx).all():
                """full grad """
                ax = sparse.csr_matrix.dot(self.A, x)
                grad = sparse.csr_matrix.dot(self.A.T, ax)
            elif idx.size == 0:
                """ empty idx """
                return np.zeros(x.shape), None
            else:
                """ batch grad """
                if idx.size == 1:
                    idx = int(idx)
                    d1 = self.A.data[self.A.indptr[idx]:self.A.indptr[idx + 1]]
                    i1 = self.A.indices[self.A.indptr[idx]:self.A.indptr[idx + 1]]
                    a_dense = np.zeros(x.size)
                    a_dense[i1] = d1
                    ax = np.dot(a_dense, x)
                    grad = np.dot(a_dense.T, ax)
                else:
                    a1 = self.A[idx, :]
                    ax = sparse.csr_matrix.dot(a1.T, x)
                    grad = sparse.csr_matrix.dot(a1.T, ax)

            if return_objval is False:
                return -grad, None
            else:
                return -grad, -0.5 * np.sum(np.trace(np.matmul(ax.T, ax)))


if __name__ == '__main__':
    import numpy.linalg as la
    import time

    sparse_mat = False
    num_instances, dimension = 1*10**3, 3*10**2
    model = Quadratic(num_instances, dimension, sparse_mat=sparse_mat)
    model.synthetic_data()
    batch_size = 1   # 'full'
    print('batch size:', batch_size)
    Epoch = 100
    tol = 1e-16
    stepsize_type = ['1/k', '1/sqrtk', 'constant']

    step_size_t = stepsize_type[2]
    if isinstance(batch_size, int):
        # 50 * batch_size / (num_instances * Epoch * np.sqrt(dimension))
        step_init = 0.001
    else:
        step_init = 0.0001

    np.random.seed(2021)
    x = np.random.randn(dimension, 1)  # random initialization
    print(la.norm(x))
    obj_val = []
    grad_norm = []
    time_record = []

    def objective(x, idx, return_objval=True):
        return model.obj(x=x, idx=idx, return_objval=return_objval)

    start_time = time.time()
    step_size = step_init
    for i in range(Epoch):

        if batch_size == 'full':
            """ full gradient """
            idx = np.arange(num_instances)
            grad, fval = objective(x, idx=idx)
            step_size_old = step_size

            if step_size_t == '1/sqrtk':
                step_size = step_init / np.sqrt(i)
            elif step_size_t == '1/k':
                step_size = step_init / i
            elif step_size_t == 'constant':
                step_size = step_init

            if step_size < tol:
                break
            # print(step_size/step_size_old)
            x -= step_size*grad
        else:
            """ stochastic gradient step, one epoch """
            if step_size_t == '1/sqrtk':
                step_size = step_init / np.sqrt(i)
            elif step_size_t == '1/k':
                step_size = step_init / i
            elif step_size_t == 'constant':
                step_size = step_init

            if step_size < tol:
                break
            num = np.int(np.ceil(num_instances / batch_size))
            sample = np.random.permutation(num_instances)
            end = 0
            for _ in range(num):
                # idx = np.random.randint(0, num_instances, batch_size)
                start = end
                end = start + batch_size
                idx = sample[start: end]
                grad, objective_val = objective(x, idx=idx)
                x -= step_size * grad

            grad, fval = objective(x, idx=np.arange(num_instances), return_objval=True)
        # print(step_size)
        time_record.append(time.time() - start_time)
        # x = x - np.power(0.9, i)*subg
        obj_val.append(fval)
        grad_norm.append(la.norm(grad))

    la.norm(x)
    print('time', time.time() - start_time)
    print({'time': time_record})
    print({'obj val': obj_val})

    import matplotlib.pyplot as plt

    plt.plot(obj_val)
    plt.yscale('log', base=10)
    plt.xlabel('Epoch')
    plt.ylabel(r'$f( {X}_k) - f^*$')
    plt.show()
