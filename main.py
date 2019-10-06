'''
Project name

Description:

Author: Charles Zhou
Date: 2019-10-05
'''


import numpy as np
from global_config import INPUT_SHAPE, RADIUS
from task_env import noisy_circle,  iou, draw_circle
from build_models import euclidean_distance_loss
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({
    "euclidean_distance_loss": euclidean_distance_loss,
})

np.random.seed(None)
from keras.models import load_model
center_predictor_name = "c_center.h5"
radius_predictor_name = 'c_radius.h5'
center_predictor = load_model(center_predictor_name)
radius_predictor = load_model(radius_predictor_name)
def main():
    size = INPUT_SHAPE[0]
    num_trials = 30
    samples = []
    for _ in range(num_trials):
        results = []
        for _ in range(100):
            params, img = noisy_circle(size, RADIUS, 2)
            params = list(params)
            detected_center = center_predictor.predict([
                np.array([np.expand_dims(img, -1)]),
                np.array([np.expand_dims(img, -1)]),
            ]
            )[0]
            detected_radius = radius_predictor.predict([
                np.array([np.expand_dims(img, -1)]),
                np.array([np.expand_dims(img, -1)]),
            ])[0]
            detected = [detected_center.tolist()[0], detected_center.tolist()[1]]+detected_radius.tolist()
            ret = iou(params, detected)
            results.append(ret)
        results = np.array(results)
        precision = (results>0.7).mean()
        samples.append(precision)
    samples = np.array(samples)
    bs_ret = bs.bootstrap(samples, stat_func=bs_stats.mean)
    print(bs_ret)

if __name__ == "__main__":
    # import tensorflow as tf

    # with tf.device('/cpu:0'):
    #     main()
    main()