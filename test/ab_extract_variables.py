'''
@author: Faizan-Uni-Stuttgart

26 May 2020

14:41:57

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\fftmasa')

    main_dir /= r'test_fftmasa_idxs_sclr_04_ppt'

    os.chdir(main_dir)

    h5_file = Path(r"fftmasa.h5")

    out_dir = Path(h5_file.parents[0].stem) / 'data_extracted'

    out_dir.mkdir(exist_ok=True)

    h5_hdl = h5py.File(h5_file, 'r')

    # Reference realization.
    ref_data = h5_hdl['data_ref/data_ref_rltzn'][...]

    ref_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

    np.savetxt(
        out_dir / 'ref_data.csv',
        ref_data,
        fmt='%0.3f',
        delimiter=';',
        header=';'.join(ref_labels),
        comments='')

    ref_data = None

    # Simulations
    sim_grp = h5_hdl['data_sim_rltzns']
    for sim_lab in sim_grp:
        sim_data = sim_grp[f'{sim_lab}/data'][...]

        np.savetxt(
            out_dir / f'sim_data_{sim_lab}.csv',
            sim_data,
            fmt='%0.3f',
            delimiter=';',
            header=';'.join(ref_labels),
            comments='')

        sim_data = None

    h5_hdl.close()
    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()