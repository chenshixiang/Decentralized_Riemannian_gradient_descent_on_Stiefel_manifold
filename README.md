# Decentralized_Riemannian_gradient_descent_on_Stiefel_manifold
Python implementation of the decentralized Riemannian stochastic(DRSGD) and decentralized Riemannian gradient tracking (DRGTA) in the paper:
  
@article{chen2021decentralized, 
  title={Decentralized Riemannian Gradient Descent on the Stiefel Manifold},  
  author={Chen, Shixiang and Garcia, Alfredo and Hong, Mingyi and Shahrampour, Shahin}, 
  journal={arXiv preprint arXiv:2102.07091},  
  year={2021} 
}
  
Please cite this paper if you use this code in your published research project. 



# Prerequisites:
python 3.6+   
numpy   
matplotlib  
mpi4py 3.0.3   
networkx 2.5	  
pickle  

# Files:  
algs: the source codes for DRGTA and DRSGD  
demo_DRGTA.py and demo_DRSGD.py:  examples showing how to use these two algorithms for solving the decentralized PCA problem

# Usage
The demos show how to use the two algorithms to solve the decentralized PCA problem.	
To run on 8 cores/devices of DRSGD algorithm, you can use the following command:  

	mpiexec -n 8 python -m mpi4py demo_DRSGD.py
  
To run on 8 cores/devices of DRGTA algorithm, you can use the following command:  

	mpiexec -n 8 python -m mpi4py demo_DRGTA.py 



# Results Explanation 
Epoch:  number of iteration 
Total CPU time(including compute average using All_Reduce)  
Local total CPU time(exclude All_Reduce and computation on mean) 	   
Consensus time: total  time of consensus iteration     
Local obj function time: time of computing local gradient and objetive value     
Local retraction time: time of computing local retraction    
Projection time:  time of computing projection onto tangent space     
MPI (All)Reduce time: CPU time of MPI Reduce and Allreduce time when the average point is computed     
time of computation on mean : CPU time of computing  gradient and objetive value at average point       
Consensus_error:  the consensus error, Frobenius norm     
Riemannian grad norm at manifold average..     
Objective val: objective value at average point      
Distance to ground truth..     
  
The results are stored in pkl files.  

# Support Euclidean space problem
This code also supports Euclidean space problem. You only need to set the manifold to Euclidean, which is defined in [ManifoldToolbox](./misc/ManifoldToolbox.py)

If you have any questions or find any bugs, feel free to contact Shixiang Chen chenshxiang@gmail.com.


