import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


def plot(log, objval=False):
    # plt.ion()
    plt.figure(1)
    plt.plot(log.consens_error)
    plt.yscale('log', base=10)
    plt.xlabel('Epoch')
    plt.ylabel(r'$ || \mathbf{x}_k - \mathbf{\bar{x}}_k ||$')
    plt.title('consensus loss v.s. iteration')
    plt.show()

    plt.figure(2)
    plt.plot(log.ave_grad_norm)
    plt.yscale('log', base=10)
    plt.xlabel('Epoch')
    plt.ylabel(r'$|| \mathrm{grad} f( {\bar{x}}_k) ||$')
    plt.title('gradient v.s. iteration')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.show()

    if objval:
        plt.figure(3)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        if log.opt_objval is not None:
            plt.plot(np.array(log.objval_ave) - log.opt_objval)
            plt.ylabel(r'$f( {\bar{x}}_k) - f^*$')
        else:
            plt.plot(np.array(log.objval_ave) -min(log.objval_ave))
            plt.ylabel(r'$f( {\bar{x}}_k) - \min_k f( {\bar{x}}_k) $')
        plt.yscale('log', base=10)
        plt.xlabel('Epoch')
        plt.title('function optimality v.s. iteration')
        plt.show()

    if log.opt_variable is not None:
        plt.figure(4)
        plt.plot(log.distance_to_opt)
        plt.yscale('log', base=10)
        plt.xlabel('Epoch')
        plt.ylabel(r'$d_s({\bar{x}}_k, x^*)$')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.title('distance to optimal set v.s. iteration')

        plt.show()
