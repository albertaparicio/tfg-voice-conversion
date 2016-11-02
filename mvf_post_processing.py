import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

from error_metrics import RMSE

# TODO General code cleanup

# returns a compiled model
# identical to the previous one
model = load_model('mvf_dnn_model.h5')

model.layers[0].W = None
model.layers[0].b = None
model.layers[3].W = None
model.layers[3].b = None
model.layers[6].W = None
model.layers[6].b = None

src_train_frames_original = np.loadtxt('mvf_src_train_frames.csv', delimiter=',')
src_train_frames = np.loadtxt('mvf_src_train_frames_nomean.csv', delimiter=',')

trg_train_frames_original = np.loadtxt('mvf_trg_train_frames.csv', delimiter=',')
trg_train_frames = np.loadtxt('mvf_trg_train_frames_nomean.csv', delimiter=',')

src_train_mean = np.mean(src_train_frames_original[:, 0], axis=0)
src_train_std = np.std(src_train_frames_original[:, 0], axis=0)

trg_train_mean = np.mean(trg_train_frames_original[:, 0], axis=0)
trg_train_std = np.std(trg_train_frames_original[:, 0], axis=0)

# RMSE of training data
prediction = model.predict(src_train_frames)

prediction[:, 0] = (prediction[:, 0] * trg_train_std) + trg_train_mean
# src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

# Mask prediction data
vf_predictions = prediction[:, 0]
uv_predictions = np.round(prediction[:, 1])
vf_predictions[np.where(uv_predictions == 0)] = 1000

# Mask ground truth data
vf_gtruth = trg_train_frames_original[:, 0]
uv_gtruth = np.round(trg_train_frames_original[:, 1])
vf_gtruth[np.where(uv_gtruth == 0)] = 1000

rmse_training = RMSE(trg_train_frames_original[:, 0], prediction[:, 0], mask=trg_train_frames_original[:, 1])

# RMSE of test data
test_data = np.loadtxt('data/test_datatable.csv.gz', delimiter=',')
vf_test_gtruth = (test_data[:, 41] - src_train_mean) / src_train_std

prediction_test = model.predict(np.column_stack((vf_test_gtruth, test_data[:, 42])))

prediction_test[:, 0] = (prediction_test[:, 0] * trg_train_std) + trg_train_mean
# src_train_frames[:, 0] = (src_train_frames[:, 0] * src_train_std) + src_train_mean

# Mask prediction data
vf_test_predictions = prediction_test[:, 0]
uv_test_predictions = np.round(prediction_test[:, 1])
vf_test_predictions[np.where(uv_test_predictions == 0)] = 1000

rmse_test = RMSE(test_data[:, 41], prediction_test[:, 0], mask=test_data[:, 42])

# np.savetxt('rmse.csv', rmse_res, delimiter=',')

print(rmse_training)
print(rmse_test)

# Histogram of predicted training data and training data itself
plt.hist(vf_predictions, bins=100)
plt.title('Prediction frames')
plt.savefig('prediction_hist.png', bbox_inches='tight')
plt.show()

# Histogram of training samples
plt.figure()
plt.hist(vf_gtruth, bins=100)
plt.title('Training target frames')
plt.savefig('gtruth_hist.png', bbox_inches='tight')
plt.show()

exit()
