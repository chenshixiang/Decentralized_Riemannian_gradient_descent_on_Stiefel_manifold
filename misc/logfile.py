import time


class Log(object):
    """
    :param start_time: Time (seconds) at which the logging process was started
    :param end_time: Time (seconds) at which the last variable was logged
    :param end_itr: Iteration at which the last variable was logged
    :param objval_ave: function value at manifold mean
    :param opt_estimate_var: Manifold mean of the variables
    :param consens_error: deviation from mean error
    :param opt_variable: optimal variable of the problem if provided
    :param opt_objval: optimal function value of the problem if provided
    :param opt_estimate_var: output variable of the algorithm
    :param opt_estimate_obj: output function value of the algorithm
    :param record_time: the cost time in recording the consensus error
    :param effective_time: total CPU time - record_time; record consensus
    :param consensus_it: communication rounds per iteration
    :param grad_stepsize:  gradient step size
    :param est_var:  local estimation variable
    :param size:  number of agents
    :param graph:  graph info
    :param time_local_obj:  CPU time of computing local gradient, obj val
    :param time_local_ret:  CPU time of computing local retraction
    :param time_reduce:  CPU time of MPI Reduce and Allreduce time
    :param time_projection:  CPU time of projection onto tangent space


    """

    def __init__(self):
        self.Algname = None
        self.data_shape = None
        self.start_time = None
        self.end_time = None
        self.end_iter = 0
        self.iter_history = []
        self.time_history = []
        self.objval_ave = []
        self.ave_grad_norm = []
        self.consens_error = []
        self.distance_to_opt = []
        self.opt_variable = None
        self.opt_objval = None
        self.opt_estimate_var = None
        self.opt_estimate_obj = None
        self.record_time = 0
        self.record_history = []
        self.effective_time = 0
        self.consensus_time = 0
        self.consensus_it = 1
        self.grad_stepsize = 0.1
        self.est_var = None
        self.size = 1
        self.graph = None
        self.time_local_obj = 0
        self.time_local_ret = 0
        self.reduce_time = 0
        self.mean_obj_time = 0
        self.time_projection = 0

    def log(self, Algname=None, data_shape=None, Iter=None, objval_ave=None, opt_var=None, opt_objval=None, ave_grad_norm=None,
            consen_error=None, distance_to_opt=None, opt_estimate_var=None, opt_estimate_obj=None, record_time=None, time_consensus=None,
            consensus_it=None, grad_stepsize=None, est_var=None, size=None, graph=None, time_local_obj=None, time_local_ret=None,
            reduce_time=None, mean_obj_time=None, time_projection=None):
        """ Log the variables, grad norm, function value with an iteration and time. """
        if Algname is not None:
            self.Algname = Algname
        if data_shape is not None:
            self.data_shape = data_shape
        if Iter is not None:
            t_now = time.time() - self.start_time
            self.iter_history.append(Iter)
            self.time_history.append(t_now)
            self.end_time = t_now
            self.end_iter = Iter
        if objval_ave is not None:
            self.objval_ave.append(objval_ave)
        if ave_grad_norm is not None:
            self.ave_grad_norm.append(ave_grad_norm)
        if consen_error is not None:
            self.consens_error.append(consen_error)
        if opt_var is not None:
            self.opt_variable = opt_var
        if opt_objval is not None:
            self.opt_objval = opt_objval
        if distance_to_opt is not None:
            self.distance_to_opt.append(distance_to_opt)
        if opt_estimate_var is not None:
            self.opt_estimate_var = opt_estimate_var
        if opt_estimate_obj is not None:
            self.opt_estimate_obj = opt_estimate_obj
        if record_time is not None:
            self.record_time += record_time
            self.record_history.append(self.record_time)
            self.effective_time = self.end_time - self.record_time
        if time_consensus is not None:
            self.consensus_time = time_consensus
        if consensus_it is not None:
            self.consensus_it = consensus_it
        if grad_stepsize is not None:
            self.grad_stepsize = grad_stepsize
        if est_var is not None:
            self.est_var = est_var
        if size is not None:
            self.size = size
        if graph is not None:
            self.graph = graph

        if time_local_obj is not None:
            self.time_local_obj = time_local_obj
        if time_local_ret is not None:
            self.time_local_ret = time_local_ret
        if reduce_time is not None:
            self.reduce_time = reduce_time
        if mean_obj_time is not None:
            self.mean_obj_time = mean_obj_time
        if time_projection is not None:
            self.time_projection = time_projection

    def print_rgd_value(self):
        if self.effective_time is None:
            self.effective_time = self.end_time

        print('====================  Results  ========================')
        print(f'Epoch: {self.end_iter};\n'
              f'Total CPU time(including compute average using All_Reduce): {self.end_time:.3f};\n'
              f'Local total CPU time(exclude All_Reduce and computation on mean): {self.effective_time:.3f};\n'
              f'Consensus time: {self.consensus_time:.3f};\n'
              f'Local obj function time: {self.time_local_obj:.3f};\n'
              f'Local retraction time: {self.time_local_ret:.3f};\n'
              f'Projection time: {self.time_projection:.3f};\n'
              f'MPI (All)Reduce time: {self.reduce_time:.3f};\n'
              f'time of computation on mean : {self.mean_obj_time:.3f};\n')
        if self.consens_error:
            print(f'Consensus_error: {self.consens_error[-1]: .4e}.')
        if self.ave_grad_norm:
            print(f'Riemannian grad norm at manifold average: {self.ave_grad_norm[-1]:.3e}')

        if self.opt_estimate_obj:
            print(f'Objective val: {self.opt_estimate_obj}')
        if self.distance_to_opt:
            print(f'Distance to ground truth: '
                  f'{min(self.distance_to_opt):.3e}')
        print('\n')
