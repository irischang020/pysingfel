import numpy as np
import h5py

with h5py.File('test_saveHDF5_parallel_intens_combined.h5', 'r') as f:  
    data1 = f['imgIntens']
    data2 = f['orientation']
    print (data1)
