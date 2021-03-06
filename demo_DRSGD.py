from mpi4py import MPI
import numpy as np

from misc.Graph import Graph
from algs.decentralized_RGD import DecenRiemannianGradientStochasticDescent
from misc.ManifoldToolbox import StiefelManifold  # set manifold
from Quadratic_object.Quadraic import Quadratic
from algs.run_algorithm import demo


comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
"""
Demo of   decentralized  Riemannian stochastic  gradient descent for solving
            min_X  sum_{i=1}^n f_i(x_i)
            s.t.  x_1 = x_2 = ... x_n, and and x_i in manifold
        where N is the number of devices
        f_i(x_i) = 0.5* trace(  (A_i x_i)^T  A_i x_i )
        A is data matrix with size of (row, col) = (matrix_row_n , matrix_col_n) =  (size*100, 10)
        A_i is obtained by divided A into N partitions
        x_i's size: (matrix_col_n, var_col_n) = (10, 2)
"""
'''==================================================================='''
""" 
set data size
data matrix A_i size: (sample_size, var_dim)
variable size: (var_dim, var_col)
"""
sample_size, var_dim, var_col = size*100, 10**1, 2
eigengap = 0.8
""" 
 set graph, in this demo, we use the ring graph, and the weighted rule is given by metropolis_hastings
"""
graph_type = ['Ring', 'ER', 'star', 'complete']
weighted_type = ['Laplacian-based', 'lazy_metropolis', 'metropolis_hastings']
Er_probability = 0.3
graph_set = (graph_type[0], weighted_type[2], Er_probability)


""" initial point """
np.random.seed(seed=1)
x_start = np.random.randn(var_dim, var_col)
x_start = StiefelManifold.proj_manifold(x_start)

""" termination """
max_iter, tol = 2*10**2, 1e-5

batch_size = 1
# batch_size = 'full'  # if the full gradient is computed
""" multi-step consensus """
T_1, T_2 = 1, 10
"""stepsize"""
"""
set stepsize 
    if stepsize_type == '1/k':
        stepsize = grad_stepsize[0]/iteration
    if stepsize_type == '1/sqrtk':
        stepsize = grad_stepsize[0]/sqrt(iteration)
    if stepsize_type == 'constant':
        stepsize = grad_stepsize[0]  
"""
stepsize_type = ['1/k', '1/sqrtk', 'constant']

# beta_0_large = 0.1
# large_step = beta_0_large / pow(max_iter, 0.5)
beta_0_small = 0.05
small_step = beta_0_small / (pow(max_iter, 0.5))

""" 
run demo of quadratic minimization on the Stiefel manifold 
"""
if rank == 0:
    """"create  global model """
    global_model = Quadratic(sample_size, var_dim, eigengap=eigengap)
    global_model.synthetic_data()
    print('==========================   New case  =============================')
    print("data matrix shape:", sample_size, ";", "variable shape:", (var_dim, var_col))
    graph = Graph(graph_type=graph_set[0], weighted_type=graph_set[1],
                  N=size, p=graph_set[2], plot_g=False)
    print("The peers of graph:", graph.peer)
else:
    global_model = None
    graph = None


""" decentralized algorithms """
import os
files_name = 'DRSGD_results'
if rank == 0:
     if not os.path.isdir(files_name):
         os.makedirs(files_name)

""" decentralized algorithms """
save_file_name = os.path.join(files_name, 'DRSGD_t_'+ str(T_1) + '_beta_' + str(beta_0_small) + '_small_stepsize.pkl')
demo(Alg=DecenRiemannianGradientStochasticDescent,
     global_model=global_model,
     f_obj=Quadratic,
     x_start=x_start,
     data_size=(sample_size, var_dim, var_col, eigengap),
     graph_setting=graph_set,
     graph=graph,
     manifold=StiefelManifold,
     consensus_stepsize=1,
     grad_stepsize=small_step,
     step_size_type='constant',
     multi_step_consen=T_1,
     batch_size=batch_size,
     termination_cond=(max_iter, tol),
     comp_objval=True,
     stop_by_time=False,
     record_consensus_error=True,
     plot=True,
     filename=save_file_name)


# demo(Alg=DecenRiemannianGradientStochasticDescent,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      step_size_type='constant',
#      multi_step_consen=T_1,
#      batch_size=batch_size,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRSGD_'+ str(T_1) + '_large_stepsize.pkl')
#
# demo(Alg=DecenRiemannianGradientStochasticDescent,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      step_size_type='constant',
#      grad_stepsize=small_step,
#      multi_step_consen=T_2,
#      batch_size=batch_size,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRSGD_'+ str(T_2) + '_small_stepsize.pkl')
#
# demo(Alg=DecenRiemannianGradientStochasticDescent,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      step_size_type='constant',
#      multi_step_consen=T_2,
#      batch_size=batch_size,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRSGD_'+ str(T_2) + '_large_stepsize.pkl')
#
#
# """ complete graph, equally weighted """
# graph_set_1 = (graph_type[-1], weighted_type[0], Er_p)
# if rank == 0:
#     graph1 = Graph(graph_type=graph_set_1[0], weighted_type=graph_set[1],
#                   N=size, p=graph_set[2], plot_g=False)
#     print("The peers of graph:", graph.peer)
# else:
#     global_model = None
#     graph1 = None
#
# demo(Alg=DecenRiemannianGradientStochasticDescent,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set_1,
#      graph=graph1,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      step_size_type='constant',
#      multi_step_consen=1,
#      batch_size=batch_size,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRSGD_complete_consensus_large_stepsize.pkl')
