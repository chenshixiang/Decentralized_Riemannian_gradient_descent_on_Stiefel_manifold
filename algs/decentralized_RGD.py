import numpy as np
from mpi4py import MPI

from misc.ManifoldToolbox import subspace_distance
from misc.Time_decorator import timeit, timeit_local_obj, timeit_local_retraction, timeit_local_projection

""" Decentralized Riemannian stochastic gradient method on manifold. """
__license__ = 'MIT'
__author__ = 'Shixiang Chen'
__email__ = 'chenshxiang@gmail.com'

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


class DecenRiemannianGradientStochasticDescent:
    """
    The decentralized Riemannian GD.

        #:param
                 CONS:  consensus object
                 local_model: local model
                 manifold: the type of manifold, which is embedded in Euclidean space
                 arg_start: the initial point
                 synch: use MPI blocking communication
                 weight: stochastic vector of the graph
                 step_size_consensus: stepsize of consensus
                 stepsize_type: type of stepsize for gradient step
                 step_size_grad: stepsize of gradient step
                 termination_condition: stopping criterion
                 record_consensus_error:  to get the consensus error, but this would be slow when epoch No. is large.
                 comp_objval: whether compute the objective value at each iteration
                 time_local_obj:  time of computing local gradient and obj val
                 time_local_ret: time of retraction at local update
                 time_local_proj: time of projection onto tangent space
     """
    def __init__(self, CONS,
                 local_model,
                 manifold,
                 arg_start,
                 synch=True,
                 weight=[],
                 batch_size=False,
                 step_size_consensus=1,
                 step_size_type='constant',
                 step_size_grad=[1, 1],
                 terminate_by_time=False,
                 termination_condition=None,
                 log=None,
                 record_consensus_error=False,
                 comp_objval=False):
        """ Initialize the decentralized optimization settings. """
        self.CONS = CONS
        self.local_model = local_model
        self.manifold = manifold
        self.arg_start = arg_start
        self.synch = synch
        self.batch_size = batch_size
        self.step_size_consensus = step_size_consensus
        self.weight = weight
        self.terminate_by_time = terminate_by_time
        self.termination_condition = termination_condition
        self.log = log
        self.step_size_type = step_size_type
        self.step_init = step_size_grad
        # self.step_gamma = step_size_grad[1]
        self.start_time = MPI.Wtime()
        self.record_consensus_error = record_consensus_error
        self.comp_objval = comp_objval
        self.time_local_obj = 0
        self.time_local_ret = 0
        self.time_local_proj = 0

    @timeit_local_projection
    def proj_tangent(self, x, d):
        return self.manifold.proj_tangent(x, d)

    @timeit_local_retraction
    def retraction(self, x, d):
        return self.manifold.retraction(x, d)

    @timeit_local_obj
    def local_objective(self, x, idx, comp_objval=False):
        # local objective func
        return self.local_model.obj(x, idx, return_objval=comp_objval)

    @timeit
    def obj_compute_at_mean(self, x):
        """ objective function only used for the average point
           #     should return (sub)gradient and function value(optional) if comp_objval is True """
        return self.local_model.obj(x, idx=np.arange(self.local_model.num_instances), n=size, return_objval=True)

    def minimize(self):
        ni = self.local_model.num_instances
        assert isinstance(ni, int) and ni > 0, 'number of instances should be positive integer'
        Iter = 1
        step_alpha = self.step_size_consensus
        step_beta = self.step_init
        x = self.arg_start  # initial variable

        # set up log
        log = self.log
        if log:
            RGD_log = log  # Log the variable, gradient, iteration

        if not self.terminate_by_time:
            num_iter_RGD = self.termination_condition[0]
            tol = self.termination_condition[1]
            stop_criteria = Iter < num_iter_RGD
        else:
            end_time = self.start_time + self.termination_condition
            stop_criteria = MPI.Wtime() < end_time

        # Start optimization at the same time
        comm.Barrier()  # no process could pass this barrier until they all call it
        # loop of the decentralized Riemannian gradient method
        stop_condition_global = True

        while stop_criteria:
            if self.synch:
                comm.Barrier()  # synchronize
            # diminishing stepsize
            if self.step_size_type == '1/sqrtk':
                step_beta = self.step_init / np.sqrt(Iter)
            elif self.step_size_type == '1/k':
                step_beta = self.step_init / Iter
            elif self.step_size_type == 'geo_diminishing':
                step_beta = self.step_gamma * step_beta
            elif self.step_size_type == 'constant':
                step_beta = self.step_init

            if step_beta < 1e-16 and self.batch_size != 0:
                break
            x = self.gradient_step(x, step_alpha, step_beta)

            if log:
                RGD_log.log(Iter=Iter)

            """ if you want to get the consensus error, this would be slow """
            if self.record_consensus_error:
                record_start = MPI.Wtime()
                manifold_average_variable = self.CONS.compute_manifold_mean(x, self.manifold)
                consensus_error, obj_val_ave, ave_grad_norm \
                    = self.CONS.compute_at_mean(manifold_average_variable, x, self.manifold, self.obj_compute_at_mean)
                if rank == 0:
                    if ave_grad_norm < 1e-16 and consensus_error < 1e-16:
                        stop_condition_global = False
                    if RGD_log.opt_variable is not None:
                        dist_to_opt = subspace_distance(manifold_average_variable, RGD_log.opt_variable)
                        stop_condition_global = dist_to_opt > tol
                        RGD_log.log(distance_to_opt=dist_to_opt)
                stop_condition_global = comm.bcast(stop_condition_global, root=0)  # broadcast from root 0 to all roots
                if log:
                    record_time = 0
                    record_time += MPI.Wtime() - record_start
                    RGD_log.log(record_time=record_time, ave_grad_norm=ave_grad_norm, objval_ave=obj_val_ave,
                                consen_error=consensus_error, reduce_time=self.CONS.reduce_time,
                                mean_obj_time=self.CONS.mean_obj_time)

            if self.terminate_by_time is False:
                stop_criteria = Iter < num_iter_RGD
            else:
                stop_criteria = MPI.Wtime() < end_time
            stop_criteria = (stop_criteria and stop_condition_global)
            Iter += 1
        if log:
            RGD_log.log(est_var=x, time_consensus=self.CONS.consensus_time, time_local_obj=self.time_local_obj,
                        time_local_ret=self.time_local_ret, time_projection=self.time_local_proj,
                        time_communication=self.CONS.communication_time)

        """ compute the average point and average gradient """
        if not self.record_consensus_error:
            manifold_average_variable = self.CONS.compute_manifold_mean(x, self.manifold)
            consensus_error, obj_val_ave, ave_grad_norm \
                = self.CONS.compute_at_mean(manifold_average_variable, x, self.manifold, self.obj_compute_at_mean)
            if rank == 0:
                if log:
                    RGD_log.log(ave_grad_norm=ave_grad_norm, objval_ave=obj_val_ave,
                                consen_error=consensus_error)

        if rank == 0:
            print('--------------------------------------------------------')
            if log:
                RGD_log.log(opt_estimate_var=manifold_average_variable, opt_estimate_obj=obj_val_ave)
                return {"Log": RGD_log}
            else:
                print( f'Consensus time: {self.CONS.consensus_time} \n mult_step_consensus: {self.CONS.num_consensus_itr} \n')

                print(f'Epoch: {Iter};\nCPU time: {MPI.Wtime() - self.start_time:.3f};\n'
                      f'consensus_error: {consensus_error:.3e}')
                if obj_val_ave:
                    print(f'Objective val: {obj_val_ave: .5e}')
                return {"ave_variable": manifold_average_variable,
                        "consensus_error": consensus_error,
                        "ave_gradient": ave_grad_norm}
        return None

    def gradient_step(self, x, step_alpha, step_beta):
        if self.batch_size == 'full':
            """ full gradient """
            idx = np.arange(self.local_model.num_instances)
            grad, objective_val = self.local_objective(x, idx=idx)
            rgrad = self.proj_tangent(x, grad)

            # another strategy
            # """  update  variable  """
            # x = self.retraction(x,  - step_beta * rgrad)
            # # multi-step consensus
            # comm_result = self.CONS.consensus(x)
            # consensus_grad = self.proj_tangent(x, comm_result - x)
            # # """  update  variable  """
            # x = self.retraction(x, step_alpha * consensus_grad)

            # multi-step consensus
            comm_result = self.CONS.consensus(x)
            consensus_grad = self.proj_tangent(x, comm_result - x)
            # """  update  variable  """
            x = self.retraction(x, step_alpha * consensus_grad - step_beta * rgrad)
        elif self.batch_size == 0:
            """ zero size, consensus algorithm """
            comm_result = self.CONS.consensus(x)
            consensus_grad = self.proj_tangent(x, comm_result - x)
            """  update  variable  """
            x = self.retraction(x, step_alpha * consensus_grad)
        else:
            """ stochastic gradient step, one epoch """
            assert isinstance(self.batch_size, int) and self.batch_size > 0, 'batch size should be positive integer'
            num = np.int(np.ceil(self.local_model.m / self.batch_size))
            sample = np.random.permutation(self.local_model.num_instances)
            end = 0
            for _ in range(num):  # all agents should have same iterations to aviod deadlock
                start = end
                end = start + self.batch_size
                idx = sample[start: end]
                grad, objective_val = self.local_objective(x, idx=idx)
                # multi-step consensus
                comm_result = self.CONS.consensus(x)
                consensus_grad = self.proj_tangent(x, comm_result - x)
                # update Riemannian gradient
                rgrad = self.proj_tangent(x, grad)
                """  update  variable  """
                x = self.retraction(x, step_alpha * consensus_grad - step_beta * rgrad)
        return x

