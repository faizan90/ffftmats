'''
Created on Jan 3, 2022

@author: Faizan3800X-Uni
'''
import os
from multiprocessing import current_process

# Due to shitty tkinter errors.
import matplotlib.pyplot as plt
plt.switch_backend('agg')

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

from .core import FFTMASAMain, FFTMASAPlot

current_process().authkey = 'ffftmats'.encode(encoding='utf_8', errors='strict')
