import matplotlib.pyplot as plt
import numpy as np


def load_traj(traj_file):

    traj = np.loadtxt(traj_file)

    traj = [traj[i, :].reshape(4,4) for i in range(traj.shape[0])]
    traj_x = [T[0, 3] for T in traj]
    traj_y = [T[1, 3] for T in traj]
    traj_z = [T[2, 3] for T in traj]

    return traj_x, traj_y, traj_z

traj_file = "/root/data/Replica/office0/traj.txt"
pert_traj_file = "./results_eval/final_result/trajectory.txt"

traj = load_traj(traj_file)
pert_traj = load_traj(pert_traj_file)
print(traj[0])


fig, axs = plt.subplots(3, 1)

for i in range(3):
    axs[i].plot(traj[i])
    axs[i].plot(pert_traj[i])


plt.show()


