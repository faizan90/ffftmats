'''
@author: Faizan-Uni-Stuttgart

Jan 24, 2022

8:24:55 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import h5py
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftmasa')

    main_dir /= r'test_fftmasa_idxs_sclr_03_ppt'

    os.chdir(main_dir)

    h5_file = Path(r'fftmasa.h5')
    #==========================================================================

    h5_hdl = h5py.File(h5_file, 'r')

    # Reference realization.
    ref_data = h5_hdl['data_ref_rltzn/data_tfm_noise'][...]

    # Simulations
    sim_datas = []
    sim_grp = h5_hdl['data_sim_rltzns']
    for sim_lab in sim_grp:
        sim_datas.append(sim_grp[f'{sim_lab}/noise'][...])

    h5_hdl.close()

    print('ref:', ref_data.std())
    for sim_lab in range(len(sim_datas)):
        print(f'sim {sim_lab}:', sim_datas[sim_lab].std())

    axes = plt.subplots(1, 3, squeeze=False, sharex=True, sharey=True)[1]
    #==========================================================================

    axes[0, 0].scatter(
        rankdata(ref_data[:, 0]) / (ref_data.shape[0] + 1.0),
        rankdata(ref_data[:, 1]) / (ref_data.shape[0] + 1.0),
        alpha=0.01,
        c='r')

    axes[0, 0].grid()
    axes[0, 0].set_axisbelow(True)
    axes[0, 0].set_aspect('equal')

    i = 0
    axes[0, i + 1].scatter(
        rankdata(sim_datas[i][:, 0]) / (sim_datas[i].shape[0] + 1.0),
        rankdata(sim_datas[i][:, 1]) / (sim_datas[i].shape[0] + 1.0),
        alpha=0.01,
        c='k')

    axes[0, i + 1].grid()
    axes[0, i + 1].set_axisbelow(True)
    axes[0, i + 1].set_aspect('equal')

    i += 1
    axes[0, i + 1].scatter(
        rankdata(sim_datas[i][:, 0]) / (sim_datas[i].shape[0] + 1.0),
        rankdata(sim_datas[i][:, 1]) / (sim_datas[i].shape[0] + 1.0),
        alpha=0.01,
        c='k')

    axes[0, i + 1].grid()
    axes[0, i + 1].set_axisbelow(True)
    axes[0, i + 1].set_aspect('equal')
    #==========================================================================

    # axes[0, 0].scatter(ref_data[:, 0], ref_data[:, 1], alpha=0.7, c='r')
    # axes[0, 0].grid()
    # axes[0, 0].set_axisbelow(True)
    # axes[0, 0].set_aspect('equal')
    #
    # i = 0
    # axes[0, i + 1].scatter(
    #     sim_datas[i][:, 0], sim_datas[i][:, 1], alpha=0.7, c='k')
    #
    # axes[0, i + 1].grid()
    # axes[0, i + 1].set_axisbelow(True)
    # axes[0, i + 1].set_aspect('equal')
    #
    # i += 1
    # axes[0, i + 1].scatter(
    #     sim_datas[i][:, 0], sim_datas[i][:, 1], alpha=0.7, c='k')
    #
    # axes[0, i + 1].grid()
    # axes[0, i + 1].set_axisbelow(True)
    # axes[0, i + 1].set_aspect('equal')
    #==========================================================================

    plt.show()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
