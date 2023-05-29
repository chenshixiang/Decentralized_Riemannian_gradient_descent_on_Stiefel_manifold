import numpy as np
from misc.ManifoldToolbox import subspace_distance
from mpi4py import MPI
from misc.Time_decorator import timeit, timeit_local_obj, timeit_local_retraction, timeit_local_projection

""" Decentralized Riemannian Gradient Tracking method on Stiefel manifold. """
__license__ = 'MIT'
__author__ = 'Shixiang Chen'
__email__ = 'chenshxiang@gmail.com'

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


class DecenRiemannianGradientTracking:
    """
    The decentralized Riemannian tracking GD.
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

    def __init__(self,
                 CONS,
                 local_model,
                 manifold,
                 arg_start,
                 synch=True,
                 step_size_consensus=1,
                 step_size_grad=None,
                 step_size_type='constant',
                 batch_size='full',
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
        self.step_size_consensus = step_size_consensus
        self.step_size_grad = step_size_grad
        self.terminate_by_time = terminate_by_time
        self.termination_condition = termination_condition
        self.log = log
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
        consensus = self.CONS.consensus
        proj_tangent = self.proj_tangent
        retraction = self.retraction
        ni = self.local_model.num_instances
        assert isinstance(ni, int) and ni > 0

        Iter = 0
        step_alpha = self.step_size_consensus
        step_beta = self.step_size_grad
        x = self.arg_start  # initial variable

        [grad, _] = self.local_objective(x, np.arange(ni))
        # self.time_local_obj += grad_time
        rgrad = proj_tangent(x, grad)  # Riemannian gradient
        tracking_grad = rgrad   # gradient tracking variable
        rgrad_old = rgrad

        # set up log
        log = self.log
        if log:
            RGD_log = log  # Log the variable, gradient, object val, iteration, time
        if not self.terminate_by_time:
            num_iter_RGD = self.termination_condition[0]
            tol = self.termination_condition[1]
            stop_criteria = Iter < num_iter_RGD
        else:
            end_time = self.start_time + self.termination_condition
            stop_criteria = MPI.Wtime() < end_time

        stop_condition_global = True
        # Start optimization at the same time
        comm.Barrier()  # no process could pass this barrier until they all call it
        # loop of the decentralized Riemannian gradient tracking method
        while stop_criteria:
            if self.synch:
                comm.Barrier()  # synchronize

            # multi-step consensus
            comm_vector = np.append(x, tracking_grad)
            comm_result = consensus(comm_vector)

            # update consensus step
            local_average = comm_result[:x.size].reshape(self.arg_start.shape) - x
            consensus_grad = proj_tangent(x, local_average)
            # -- update RGD-tracking iteration  -- #
            v = proj_tangent(x, tracking_grad)
            x = retraction(x, step_alpha*consensus_grad - step_beta*v)
            # update Riemannian gradient
            [grad, _] = self.local_objective(x, np.arange(ni))

            rgrad = proj_tangent(x, grad)
            # update gradient tracking estimate
            tracking_grad = comm_result[x.size:].reshape(self.arg_start.shape)
            tracking_grad += rgrad - rgrad_old

            # Update the gradient
            rgrad_old = rgrad
            # log the iteration number
            if log:
                RGD_log.log(Iter=Iter)
            """ compute the consensus error as termination criterion, but this would be slow """
            if self.record_consensus_error:
                record_start = MPI.Wtime()
                manifold_average_variable = self.CONS.compute_manifold_mean(x, self.manifold)
                consensus_error, obj_val_ave, ave_grad_norm \
                    = self.CONS.compute_at_mean(manifold_average_variable, x, self.manifold, self.obj_compute_at_mean)
                if rank == 0:
                    if ave_grad_norm < tol:
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
                                consen_error=consensus_error, reduce_time=self.CONS.reduce_time, mean_obj_time=self.CONS.mean_obj_time)
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
                = self.CONS.compute_at_mean(manifold_average_variable, x, self.manifold,  self.obj_compute_at_mean)

        if rank == 0:
            print('--------------------------------------------------------')
            if log:
                RGD_log.log(opt_estimate_var=manifold_average_variable, opt_estimate_obj=obj_val_ave,
                            ave_grad_norm=ave_grad_norm, objval_ave=obj_val_ave, consen_error=consensus_error)
                return {"Log": RGD_log}
            else:
                print(
                    f'Consensus time: {self.CONS.consensus_time} \n mult_step_consensus: {self.CONS.num_consensus_itr} \n')

                print(f'Iter number: {Iter};\nCPU time: {MPI.Wtime() - self.start_time:.3f};\n'
                      f'consensus_error: {consensus_error:.3e}')
                if obj_val_ave:
                    print(f'Objective val: {obj_val_ave: .5e}')
                return {"ave_variable": manifold_average_variable,
                        "consensus_error": consensus_error,
                        "ave_gradient": ave_grad_norm}
        return None
