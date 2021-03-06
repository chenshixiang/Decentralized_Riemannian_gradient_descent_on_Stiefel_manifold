from mpi4py import MPI
import numpy as np

from misc.Graph import Graph
from algs.decentralized_RGD_tracking import DecenRiemannianGradientTracking
from misc.ManifoldToolbox import StiefelManifold  # set manifold
from Quadratic_object.Quadraic import Quadratic

from algs.run_algorithm import demo


comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
"""
The following code is the demo of   decentralized  Riemannian   gradient tracking for solving
            min_X  sum_{i=1}^n f_i(x_i)
            s.t.  x_1 = x_2 = ... x_n, and and x_i on Stiefel manifold
        where n=size is the number of devices/processes/cores
        f_i(x_i) = 0.5* trace(  (A_i x_i)^T  A_i x_i )
        A is data matrix with size of (row, col) = (matrix_row_n , matrix_col_n) = (size*100, 10)
        A_i is obtained by divided A into n partitions
        x_i's size: (matrix_col_n, var_col_n) = (10,2)
"""
'''==================================================================='''
""" 
set data size
data matrix A_i size: (sample_size, var_dim)
variable size: (var_dim, var_col)
"""
sample_size, var_dim, var_col = size*10**2, 10**1, 2
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
max_iter, tol = 10**4, 1e-8

""" multi-step consensus """
T_1, T_2 = 1, 10
"""stepsize"""
# beta_0_large = 0.05
# large_step = beta_0_large * size / sample_size
beta_0_small = 0.01
small_step = beta_0_small * size / sample_size

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
files_name = 'DRGTA_results'
if rank == 0:
     if not os.path.isdir(files_name):
         os.makedirs(files_name)

""" decentralized algorithms """
save_file_name = os.path.join(files_name, 'DRGTA_t_'+ str(T_1) + '_beta_' + str(beta_0_small) + '_small_stepsize.pkl')
# save_file_name = os.path.join(files_name, 'MPI_' + str(size)
#                               + 'DRGTA_t_'+ str(T_1) + '_beta_' + str(beta_0_small) + '_small_stepsize.pkl')
demo(Alg=DecenRiemannianGradientTracking,
     global_model=global_model,
     f_obj=Quadratic,
     x_start=x_start,
     data_size=(sample_size, var_dim, var_col, eigengap),
     graph_setting=graph_set,
     graph=graph,
     manifold=StiefelManifold,
     consensus_stepsize=1,
     grad_stepsize=small_step,
     multi_step_consen=T_1,
     termination_cond=(max_iter, tol),
     comp_objval=True,
     stop_by_time=False,
     record_consensus_error=True,
     plot=True,
     filename=save_file_name)


# demo(Alg=DecenRiemannianGradientTracking,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      multi_step_consen=T_1,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRGTA_'+ str(T_1) + '_large_stepsize.pkl')
#
# demo(Alg=DecenRiemannianGradientTracking,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=small_step,
#      multi_step_consen=T_2,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRGTA_'+ str(T_2) + '_small_stepsize.pkl')
#
# demo(Alg=DecenRiemannianGradientTracking,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set,
#      graph=graph,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      multi_step_consen=T_2,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRGTA_'+ str(T_2) + '_large_stepsize.pkl')
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
# demo(Alg=DecenRiemannianGradientTracking,
#      global_model=global_model,
#      f_obj=Quadratic,
#      x_start=x_start,
#      data_size=(sample_size, var_dim, var_col, eigengap),
#      graph_setting=graph_set_1,
#      graph=graph1,
#      manifold=StiefelManifold,
#      consensus_stepsize=1,
#      grad_stepsize=large_step,
#      multi_step_consen=1,
#      termination_cond=(max_iter, tol),
#      comp_objval=True,
#      stop_by_time=False,
#      record_consensus_error=True,
#      plot=False,
#      filename='DRGTA_complete_consensus_large_stepsize.pkl')
