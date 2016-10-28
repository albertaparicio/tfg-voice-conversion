import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

from error_metrics import RMSE

# returns a compiled model
# identical to the previous one
model = load_model('dnn_model.h5')

src_train_frames_original = np.loadtxt('src_train_frames.csv', delimiter=',')
src_train_frames = np.loadtxt('src_train_frames_nomean.csv', delimiter=',')

src_train_mean = np.mean(src_train_frames_original[:, 0], axis=0)
src_train_std = np.std(src_train_frames_original[:, 0], axis=0)

# RMSE of training data
prediction = model.predict(src_train_frames)

prediction[:, 0] = (prediction[:, 0] * src_train_std) + src_train_mean
# src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

# Mask prediction data
vf_predictions = prediction[:, 0]
uv_predictions = np.round(prediction[:, 1])
vf_predictions[np.where(uv_predictions == 0)] = 1000

# Mask ground truth data
vf_gtruth = src_train_frames_original[:, 0]
uv_gtruth = np.round(src_train_frames_original[:, 1])
vf_gtruth[np.where(uv_gtruth == 0)] = 1000

rmse_res = RMSE(np.column_stack((vf_gtruth, uv_gtruth)), np.column_stack((vf_predictions, uv_predictions)))
# np.savetxt('rmse.csv', rmse_res, delimiter=',')

print(rmse_res[0])

# Histogram of predicted training data and training data itself
plt.hist(vf_predictions, bins=100)
plt.title('Prediction frames')
plt.savefig('prediction_hist.png', bbox_inches='tight')
plt.show()

# Histogram of training samples
plt.figure()
plt.hist(vf_gtruth, bins=100)
plt.title('Training frames')
plt.savefig('gtruth_hist.png', bbox_inches='tight')
plt.show()

exit()
