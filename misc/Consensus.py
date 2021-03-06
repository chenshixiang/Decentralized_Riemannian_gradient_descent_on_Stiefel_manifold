"""
local consensus in Euclidean space,
synchronous, communication blocking model
"""

import numpy as np
import numpy.linalg as la
from mpi4py import MPI
import warnings

__license__ = 'MIT'
__author__ = 'Shixiang Chen'
__email__ = 'chenshxiang@gmail.com'

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


class EuclideanConsensus:
    """
    :param:
           synch: Whether to run the alg. synchronously (or asynchronously)
           terminate_by_time: Whether to terminate the alg. after some threshold time
           termination_condition: stopping criterion, e.g. iteration number 100, termination_condition = 100
           num_consensus_itr: number of  multi-step consensus step
           hold_on_time: maximum time to wait

    :returns:
           consensus_time: communication time except reduce
           reduce_time: Allreduce and reduce time, needed if it is required to compute consensus error
           mean_obj_time: the time of computing objective val and gradient at average point
    """

    def __init__(self, synch=True, terminate_by_time=False, num_consensus_itr=1, hold_on_time=10**3):
        self.synch = synch
        self.size = size
        self.local_data = np.empty(1)
        self.weight = []
        self.peer = []
        self.terminate_by_time = terminate_by_time
        self.num_consensus_itr = num_consensus_itr
        self.consensus_time = 0
        self.reduce_time = 0
        self.mean_obj_time = 0
        self.hold_on_time = hold_on_time
        if self.synch and self.terminate_by_time:
            warnings.warn("Use of synchronous and time term_by_time  will result in deadlocks.")

    # def send_to_neighbors(self, x):
    #     for i in self.peer:
    #         comm.Isend(x, dest=i, tag=rank)
    #
    # def collect_from_neighbors(self, x):
    #     x = x * self.weight[rank]
    #     wait_time_start = MPI.Wtime()
    #     while True:
    #         for i in self.peer:
    #             recvbuf = np.empty(x.shape, dtype=np.float64)
    #             req = comm.Irecv(recvbuf, source=i)
    #             req.wait()
    #             x += self.weight[i] * recvbuf
    #             # del status
    #         return x
    #         if MPI.Wtime() - wait_time_start > hold_on_time:
    #             warnings('wait too long for recieving, may deadlock...')

    def send_to_neighbors(self, x):
        for i in self.peer:
            comm.Send(x, dest=i, tag=rank)

    def collect_from_neighbors(self, x):
        x = x * self.weight[rank]
        # wait_time_start = MPI.Wtime()
        while True:
            for i in self.peer:
                # status = MPI.Status()
                # while not comm.Probe(source=MPI.ANY_SOURCE, status=status):
                #     pass
                recvbuf = np.empty(x.shape, dtype=np.float64)
                comm.Recv(recvbuf, source=i)
                x += self.weight[i] * recvbuf
                # del status
            return x
            # if MPI.Wtime() - wait_time_start > hold_on_time:
            #     warnings('wait too long for recieving, may deadlock...')

    def consensus(self, x):
        t0 = MPI.Wtime()
        for _ in range(self.num_consensus_itr):
            self.send_to_neighbors(x)
            x = self.collect_from_neighbors(x)
            comm.Barrier()
        self.consensus_time += MPI.Wtime() - t0
        return x

    def compute_manifold_mean(self, x, manifold):
        euclidean_average_variable = np.empty(x.shape, dtype=np.float64)
        reduce_start_time = MPI.Wtime()
        comm.Allreduce(x, euclidean_average_variable, MPI.SUM)
        self.reduce_time += MPI.Wtime() - reduce_start_time
        euclidean_average_variable = euclidean_average_variable / size
        manifold_average_variable = manifold.proj_manifold(euclidean_average_variable)
        return manifold_average_variable

    def compute_at_mean(self, manifold_average_variable, x, manifold, objective):
        local_distance_to_mean = la.norm(manifold_average_variable - x) ** 2
        reduce_start_time = MPI.Wtime()
        consensus_error = comm.reduce(local_distance_to_mean, MPI.SUM, root=0)
        if rank == 0:
            consensus_error = np.sqrt(consensus_error/size)
        self.reduce_time += MPI.Wtime() - reduce_start_time
        [local_grad_ave, local_obj_val_ave], time_it = objective(x=manifold_average_variable)
        self.mean_obj_time += time_it
        sendbuf = np.append(local_grad_ave, local_obj_val_ave)
        if rank == 0:
            recv_buf = np.empty(sendbuf.size)
        else:
            recv_buf = None
        reduce_start_time = MPI.Wtime()
        comm.Reduce(sendbuf, recv_buf, MPI.SUM, root=0)
        self.reduce_time += MPI.Wtime() - reduce_start_time

        obj_val_ave = np.empty(1, np.float64)
        ave_grad_norm = np.empty(1, np.float64)
        if rank == 0:
            grad_ave = recv_buf[:-1].reshape(x.shape) / size
            obj_val_ave = recv_buf[-1] / size
            rgrad_ave = manifold.proj_tangent(manifold_average_variable, grad_ave)
            ave_grad_norm = la.norm(rgrad_ave)
        return consensus_error, obj_val_ave, ave_grad_norm

    def partition_data_mat(self, data_mat, dimension, graph, count, remainder, csr_sparse=False):
        """ partition data into n folds """
        if not csr_sparse:
            if rank == 0:
                if rank < remainder:
                    # The first 'remainder' ranks get 'count + 1' tasks each
                    start = 0
                    stop = start + count
                else:
                    # The remaining 'size - remainder' ranks get 'count' task each
                    start = 0
                    stop = start + (count - 1)
                local_data = data_mat[start:stop + 1, :]
                print(graph.W)
                weighted = graph.W[0, :]

                # divide global_data into local data
                for i in range(1, size):
                    if i < remainder:
                        # The first 'remainder' ranks get 'count + 1' tasks each
                        start = i * (count + 1)
                        stop = start + count
                    else:
                        # The remaining 'size - remainder' ranks get 'count' task each
                        start = stop + 1
                        stop = start + (count - 1)
                    sendbuf = np.append(data_mat[start:stop + 1, :], graph.W[i, :])
                    comm.Send(sendbuf, dest=i)

            elif rank < remainder:
                recvbuf = np.empty((count + 1) * dimension + size)
                comm.Recv(recvbuf, source=0)
                local_data = recvbuf[:(count + 1) * dimension].reshape(count+1, dimension)
                weighted = recvbuf[-size:].reshape(size)
            else:
                recvbuf = np.empty(count * dimension + size)
                comm.Recv(recvbuf, source=0)
                local_data = recvbuf[:count * dimension].reshape(count, dimension)
                weighted = recvbuf[-size:].reshape(size)
            self.local_data = local_data
            self.weight = weighted
            for i in range(size):
                if self.weight[i] > 0 and i != rank:
                    self.peer.append(i)
        else:
            """  partition for csr sparse matrix:
            for the i-th row, the column indices of non-zeros are indices[indptr[i]:indptr[i+1]],
                          data is data[indptr[i]:indptr[i+1]] """
            from scipy.sparse import csr_matrix

            def find_idx(arr, begin, end):
                if arr[begin] == arr[end]:
                    return 0, 0
                return arr[begin], arr[end]

            def reform_row_num(arr, lenth):
                row_n = np.zeros(lenth)
                idx = 0
                x_old = arr[0]
                for i in range(1, lenth):
                    rep = arr[i] - x_old
                    for j in range(rep):
                        row_n[idx+j] = i-1
                    x_old = arr[i]
                    idx += rep
                    if idx >= lenth: break
                return row_n

            if rank == 0:
                csr_l = [data_mat.data, data_mat.indices, data_mat.indptr]
                if rank < remainder:
                    # The first 'remainder' ranks get 'count + 1' tasks each ( row no. start: stop+1)
                    start = 0
                    stop = start + count
                else:
                    # The remaining 'size - remainder' ranks get 'count' task each
                    start = 0
                    stop = start + (count - 1)

                ind_start, ind_end = find_idx(csr_l[2], start, stop+1)
                _data = csr_l[0][ind_start:ind_end]
                col_num = csr_l[1][ind_start:ind_end]
                row_num = reform_row_num(csr_l[2][start:stop+2], col_num.size)
                local_data = csr_matrix((_data, (row_num, col_num)), shape=(stop-start+1, dimension))
                weighted = graph.W[0, :]

                # divide global_data into local data
                for i in range(1, size):
                    if i < remainder:
                        # The first 'remainder' ranks get 'count + 1' tasks each
                        start = stop + 1
                        stop = start + count
                    else:
                        # The remaining 'size - remainder' ranks get 'count' task each
                        start = stop + 1
                        stop = start + (count - 1)

                    ind_start, ind_end = find_idx(csr_l[2], start, stop+1)
                    _data = csr_l[0][ind_start:ind_end]
                    col_num = csr_l[1][ind_start:ind_end]

                    row_num = reform_row_num(csr_l[2][start:stop+2], col_num.size)
                    # local_data = csr_matrix((_data, (row_num, col_num)), shape=(stop - start + 1, dimension))

                    comm.send(row_num.size, dest=i, tag=1)
                    sendbuf = np.concatenate((row_num, col_num, _data, graph.W[i, :]), axis=None)
                    comm.Send(sendbuf, dest=i, tag=2)

            elif rank < remainder:
                recvsize = comm.recv(source=0, tag=1)
                recvbuf = np.empty(recvsize * 3 + size)
                comm.Recv(recvbuf, source=0, tag=2)
                row_num = recvbuf[:recvsize]
                col_num = recvbuf[recvsize:2 * recvsize]
                _data = recvbuf[2 * recvsize:3 * recvsize]
                weighted = recvbuf[-size:].reshape(size)
                local_data = csr_matrix((_data, (row_num, col_num)), shape=(count + 1, dimension))
            else:
                recvsize = comm.recv(source=0, tag=1)
                recvbuf = np.empty(recvsize * 3 + size)
                comm.Recv(recvbuf, source=0, tag=2)
                row_num = recvbuf[:recvsize]
                col_num = recvbuf[recvsize:2 * recvsize]
                _data = recvbuf[2 * recvsize:3 * recvsize]
                weighted = recvbuf[-size:].reshape(size)
                local_data = csr_matrix((_data, (row_num, col_num)), shape=(count, dimension))
            self.local_data = local_data
            self.weight = weighted
            for i in range(size):
                if self.weight[i] > 0 and i != rank:
                    self.peer.append(i)


