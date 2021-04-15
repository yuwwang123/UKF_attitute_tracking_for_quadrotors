
from scipy import io
import os
import numpy as np
import matplotlib.pyplot as plt
from estimate_rot import estimate_rot, vicon_to_eulers, process_imu_raw
import click

@click.command()
@click.option('--data_id', default='2', help='dataset number', type=int)

def main(data_id):
    rolls, pitches, yaws, ang_vels, traces = estimate_rot(data_id)
    vicon = io.loadmat('vicon/viconRot'+str(data_id)+'.mat')
    _, vicon_eulers = vicon_to_eulers(vicon)

    imuRaw = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_id) + ".mat"))
    imu = imuRaw['vals']
    # gyros = imug_cal.T
    _, gyros = process_imu_raw(imu[0:3, :].view(np.int16),
                               imu[3:6, :].view(np.int16))


    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(vicon_eulers[:, 0], label="ground truth")
    axs[0].plot(rolls, label="ukf estimate")
    axs[0].set_title('roll')
    axs[0].legend(loc="upper right", shadow=True, fancybox=True)

    axs[1].plot(vicon_eulers[:, 1], label="ground truth")
    axs[1].plot(pitches, label="ukf estimate")
    axs[1].set_title('pitch')
    axs[1].legend(loc="upper right", shadow=True, fancybox=True)


    axs[2].plot(vicon_eulers[:, 2], label="ground truth")
    axs[2].plot(yaws, label="ukf estimate")
    axs[2].set_title('yaw')
    axs[2].legend(loc="upper right", shadow=True, fancybox=True)

    plt.show()

if __name__ == '__main__':
    main()