import pickle
import matplotlib.pyplot as plt
import os

switcher = {
    'DecenRiemannianGradientStochasticDescent': 'DRSGD',
    'DecenRiemannianGradientTracking': 'DRGTA'
}

line_stype = ['-2', '-*', '--+', '--o', '-1']


def plot_dist_to_opt(log, curve_type, size=5, interval=10, linewidth=3):
    t, beta = log.consensus_it, log.grad_stepsize
    beta_0 = beta * pow(log.end_iter, 0.5)
    print(log.Algname)
    alg_name = switcher.get(log.Algname, "Invalid algorithm")
    if log.opt_variable is not None:
        plt.figure(1)
        fig = plt.plot(log.distance_to_opt, curve_type, linewidth=linewidth, markersize=size, markevery=len(log.distance_to_opt)//interval,
                       label= alg_name + ', t=' + str(t) + ', ' + r'$\hat{\beta}$=' + str(beta_0))
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel(r'$d_s({\bar{x}}_k, x^*)$')

        # plt.ylabel(r'$d_s({\bar{x}}_k, x^*)$', fontsize=14)


import os
files_name = 'DRSGD_results'

""" decentralized algorithms """
save_file_name = os.path.join(files_name, 'DRSGD_t_'+ str(1) + '_beta_' + str(0.05) + '_small_stepsize.pkl')

with open(save_file_name, 'rb') as input:
    log = pickle.load(input)
    curve_type = line_stype[0]
    plot_dist_to_opt(log, curve_type, size=10, interval=15, linewidth=1)


# with open('DRSGD_1_large_stepsize.pkl', 'rb') as input:
#     log = pickle.load(input)
#     curve_type = line_stype[1]
#     plot_dist_to_opt(log, curve_type, interval=15, linewidth=1)
#
# with open('DRSGD_10_small_stepsize.pkl', 'rb') as input:
#     log = pickle.load(input)
#     curve_type = line_stype[2]
#     plot_dist_to_opt(log, curve_type)
#
#
# with open('DRSGD_10_large_stepsize.pkl', 'rb') as input:
#     log = pickle.load(input)
#     curve_type = line_stype[3]
#     plot_dist_to_opt(log, curve_type)
#
# with open('DRSGD_complete_consensus_large_stepsize.pkl', 'rb') as input:
#     log_central = pickle.load(input)
#     beta_0 = log_central.grad_stepsize * pow(log_central.end_iter, 0.5)
#     plt.yscale('log')
#     plt.xlabel('Epoch')
#     # plt.ylabel(r'$d_s({\bar{x}}_k, x^*)$')
#     plt.plot(log_central.distance_to_opt, line_stype[4], linewidth=2,
#              markevery=len(log_central.distance_to_opt)//10,
#              label='complete, ' + r'$\hat{\beta}=$' + str(beta_0))

large_font = 15
plt.rc('xtick', labelsize=20)    # fontsize of the x and y labels
plt.xticks(fontsize=large_font)
plt.rc('ytick', labelsize=20)
plt.yticks(fontsize=large_font)

plt.rc('legend', fontsize=14)    # legend fontsize

plt.legend()
graph_type, weighted_rule, prob = log.graph

if not os.path.isdir("DRSGD_synthetic"):
    os.makedirs("DRSGD_synthetic")
save_path = 'DRSGD_synthetic'
filename = 'DRSGD_shape_' + str(log.data_shape) + '_nodes_' + str(log.size) + '_p_' + str(prob)+ '_'+ graph_type +'_'+ weighted_rule  +'.pdf'
plt.savefig(os.path.join(save_path, filename), format='pdf', dpi=3000)
plt.show()
