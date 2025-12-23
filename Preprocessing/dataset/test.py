import numpy as np

data = np.load('./ACCESS-CM2_standardized/ssp245_train.npz')['sst']
print('Train batch shape:', data[:4].shape)
print('NaN:', np.isnan(data[:4]).any())
print('Inf:', np.isinf(data[:4]).any())
print('min/max:', np.nanmin(data[:4]), np.nanmax(data[:4]))
