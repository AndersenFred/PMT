import pyvisa as py
import numpy as np
import h5py 
from tqdm import tqdm_notebook

OPEN_CMD =  'TCPIP0::{}::INSTR'

