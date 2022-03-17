'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

from scipy.interpolate import interp1d

from gnrctsgenr import (
    GTGBase,
    GTGData,
    GTGPrepareBase,
    GTGPrepareCDFS,
    GTGPrepareUpdate,
    GTGAlgBase,
    GTGAlgObjective,
    GTGAlgIO,
    GTGAlgTemperature,
    GTGAlgMisc,
    GTGAlgorithm,
    GTGSave,
    )

from .aa_setts import FFTMASASettings

from .ba_prep import (
    FFTMASAPrepareRltznRef,
    FFTMASAPrepareRltznSim,
    FFTMASAPrepareTfms,
    FFTMASAPrepare,
    )

from .ca_alg import (
    FFTMASAAlgLagNthWts,
    FFTMASAAlgLabelWts,
    FFTMASAAlgAutoObjWts,
    )

from .da_rltzn import FFTMASARealization


class FFTMASAMain(
        GTGBase,
        GTGData,
        FFTMASASettings,
        GTGPrepareBase,
        FFTMASAPrepareTfms,
        GTGPrepareCDFS,
        GTGPrepareUpdate,
        FFTMASAPrepare,
        GTGAlgBase,
        GTGAlgObjective,
        GTGAlgIO,
        FFTMASAAlgLagNthWts,
        FFTMASAAlgLabelWts,
        FFTMASAAlgAutoObjWts,
        FFTMASARealization,
        GTGAlgTemperature,
        GTGAlgMisc,
        GTGAlgorithm,
        GTGSave):

    def __init__(self, verbose):

        GTGBase.__init__(self, verbose)
        GTGData.__init__(self)
        FFTMASASettings.__init__(self)

        self._rr = FFTMASAPrepareRltznRef()  # Reference.
        self._rs = FFTMASAPrepareRltznSim()  # Simulation.

        GTGPrepareBase.__init__(self)
        FFTMASAPrepareTfms.__init__(self)
        GTGPrepareCDFS.__init__(self)
        GTGPrepareUpdate.__init__(self)
        FFTMASAPrepare.__init__(self)
        GTGAlgBase.__init__(self)
        GTGAlgObjective.__init__(self)
        GTGAlgIO.__init__(self)
        FFTMASAAlgLagNthWts.__init__(self)
        FFTMASAAlgLabelWts.__init__(self)
        FFTMASAAlgAutoObjWts.__init__(self)
        FFTMASARealization.__init__(self)
        GTGAlgTemperature.__init__(self)
        GTGAlgMisc.__init__(self)
        GTGAlgorithm.__init__(self)
        GTGSave.__init__(self)

        self._main_verify_flag = False
        return

    def _write_ref_rltzn_extra(self, *args):

        h5_hdl = args[0]

        ref_grp_lab = 'data_ref_rltzn'

        datas = []
        for var in self._rr.skip_io_vars:
            datas.append((var, getattr(self._rr, var)))

        ref_grp = h5_hdl[ref_grp_lab]

        for data_lab, data_val in datas:
            # Noise distribution functions.
            if (isinstance(data_val, dict) and

                  all([isinstance(key, str) for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'{data_lab}_{key}'

                    if f'{lab}_x' in ref_grp:
                        break

                    ref_grp[f'{lab}_x'] = data_val[key].x
                    ref_grp[f'{lab}_y'] = data_val[key].y

            else:
                raise NotImplementedError(
                    f'Unknown type {type(data_val)} for variable '
                    f'{data_lab}!')

        return

    def _write_sim_rltzn_extra(self, *args):

        _ = args

        # h5_hdl = args[0]
        #
        # main_sim_grp_lab = 'data_sim_rltzns'
        #
        # sim_grp_lab = self._rs.label
        #
        # sim_grp_main = h5_hdl[main_sim_grp_lab]
        #
        # sim_grp = sim_grp_main[sim_grp_lab]

        return

    def verify(self):

        GTGData._GTGData__verify(self)

        FFTMASASettings._FFTMASASettings__verify(self)

        assert self._sett_ann_fftma_sa_sett_verify_flag, (
            'FFTMA Annealing settings in an unverfied state!')

        FFTMASAPrepare._FFTMASAPrepare__verify(self)
        GTGAlgorithm._GTGAlgorithm__verify(self)
        GTGSave._GTGSave__verify(self)

        assert self._save_verify_flag, 'Save in an unverified state!'

        self._main_verify_flag = True
        return
