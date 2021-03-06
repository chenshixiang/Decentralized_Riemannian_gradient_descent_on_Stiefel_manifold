import numpy as np
import scipy.linalg as la
""" Manifold package, Stiefel manifold and Euclidean space """
__license__ = 'MIT'
__author__ = 'Shixiang Chen'
__email__ = 'chenshxiang@gmail.com'


class StiefelManifold:
    # def __int__(self):
    @staticmethod
    def proj_manifold(x):   # orthogonal projection onto the manifold
        u, s, vh = la.svd(x, full_matrices=False)
        return np.matmul(u, vh)

    @staticmethod
    def proj_tangent(x, d):
        xd = np.matmul(x.T,  d)
        pd = d - 0.5 * np.matmul(x, xd + xd.T)
        return pd

    # polar decomposition
    @staticmethod
    def retraction(x, d):
        u, s, vh = la.svd(x + d, full_matrices=False)
        return np.matmul(u, vh)


class Euclidean:
    @staticmethod
    def proj_manifold(x):  # orthogonal projection onto the manifold
        return x

    @staticmethod
    def proj_tangent(x, d):
        return d

    # polar decomposition
    @staticmethod
    def retraction(x, d):
        return x + d


def subspace_distance(x, y):
    """ distance between two subspaces: [x] and [y]
        distance = min_{R in O(d)} || x  - yR^T ||^2
    """
    xty = np.matmul(x.T, y)
    if xty.size == 1:
        R = np.sign(xty)
        return la.norm(x - y*R)
    else:
        u, s, vh = la.svd(xty)
        R = np.matmul(u, vh)
        return la.norm(x - np.matmul(y, R.T))


def subspace_distance_nonorthogonal(x, y):
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    v, _, _ = np.linalg.svd(y, full_matrices=False)
    return subspace_distance(u, v)


# if __name__ == '__main__':
#     # test
#     M = StiefelManifold()
#     n, r = 10, 2
#     x = np.random.randn(n, r)
#     u, s, vh = la.svd(x, full_matrices=False)
#     print(type(x[0, 0]))
#     x = np.matmul(u, vh)
#     d = np.random.randn(n, r)
#     px_d = M.proj_tagent(x, d)
#     print(la.norm(np.dot(x.T, px_d) + np.dot(px_d.T, x)))
#     print(np.allclose(np.zeros(r), np.dot(x.T, px_d) + np.dot(px_d.T, x)))
#     y = M.retraction(x,d)
#     print(np.allclose(np.identity(r), np.dot(y.T, y)))
