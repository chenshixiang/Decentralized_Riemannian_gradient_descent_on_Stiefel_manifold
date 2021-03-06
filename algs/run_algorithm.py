import numpy as np
from mpi4py import MPI
from misc.logfile import Log

from misc.ManifoldToolbox import StiefelManifold  # set manifold
from misc.Consensus import EuclideanConsensus

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


def demo(Alg, global_model, f_obj, x_start, data_size, graph_setting, graph, manifold=StiefelManifold, consensus_stepsize=1,
         grad_stepsize=0,   step_size_type='constant', multi_step_consen=1,  batch_size='full', stop_by_time=False,
         termination_cond=(1e2, 1e-8), plot=True, comp_objval=True, record_consensus_error=True, filename=None):
    """
    Demo of The decentralized Riemannian tracking GD.
       #:param data_size = (matrix_row_num, matrix_col_num, var_col_num)
       #:param graph_setting: network graph setting
       #:param manifold: the type of manifold
       #:param consensus_stepsize: stepsize of consensus
       #:param multi_step_consen: number of consensus iteration
       #:param grad_stepsize: stepsize of gradient step
       #:param stop_by_time: whether stop by CPU time
       #:param termination_cond=(iteration number , tol): stopping criterion,
       #:param plot: plot the objective values and gradient norm if plot is True and log_ is True
       #:param comp_objval: whether compute objective value
       #:param record_consensus_error:  to get the consensus error, but this would be slow.
    """

    num_instances = data_size[0]
    dimension = data_size[1]
    count = num_instances // size
    remainder = num_instances % size

    if rank < remainder:
        ni = count + 1
    else:
        ni = count

    CONS = EuclideanConsensus(synch=True,
                              terminate_by_time=stop_by_time,
                              num_consensus_itr=multi_step_consen)
    start_time = MPI.Wtime()
    DRG_log = Log()
    if rank == 0:
        global_data = global_model.A
        print('================= ' + Alg.__name__ + ' ================= ')
        print("data matrix shape:", global_data.shape, ";", "variable shape:", (data_size[1], data_size[2]))

        if global_model.x_true is not None:
            x_true = global_model.x_true[:data_size[2], :].T
            Ax = np.matmul(global_data, x_true)
            opt_val = -0.5*np.sum(np.matmul(Ax.T, Ax))/size
            print('optimal val', opt_val)
            DRG_log.log(opt_objval=opt_val, opt_var=x_true)
            del global_model
        DRG_log.log(Algname=Alg.__name__, data_shape=data_size, graph=graph_setting, consensus_it=multi_step_consen, grad_stepsize=grad_stepsize, size=size)
    else:
        # DRG_log = None
        global_data = None
        graph = None

    """ partition global data into local data """
    CONS.partition_data_mat(global_data, dimension, graph, count, remainder, csr_sparse=False)
    print("rank=", rank)
    print("partition data time:", MPI.Wtime() - start_time)

    """ create local model using partitioned data """
    local_model = f_obj(num_instances=ni, dimension=dimension, col=data_size[2], m=count+1, A=CONS.local_data)
    print("local data size:", CONS.local_data.shape)
    if rank == 0:
        del global_data
    alg = Alg(local_model=local_model,
              CONS=CONS,
              manifold=manifold,
              arg_start=x_start,
              synch=True,
              step_size_consensus=consensus_stepsize,
              step_size_type=step_size_type,
              step_size_grad=grad_stepsize,
              batch_size=batch_size,
              terminate_by_time=stop_by_time,
              termination_condition=termination_cond,
              log=DRG_log,
              record_consensus_error=record_consensus_error,
              comp_objval=comp_objval)

    """ run the algorithm """
    import time
    DRG_log.start_time = time.time()
    alg_log = alg.minimize()
    if rank == 0 and alg.log:
        logfile = alg_log['Log']
        import pickle
     
        with open(filename, 'wb') as output:  # save results
            pickle.dump(logfile, output, pickle.HIGHEST_PROTOCOL)
        import time
        time.sleep(2)
        logfile.print_rgd_value()  # print result
        if plot and record_consensus_error:
            from misc.Plot import plot
            plot(logfile, objval=comp_objval)

    return alg_log

