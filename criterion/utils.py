import numpy as np
import sklearn.metrics
import tensorflow as tf

def compute_eer(label, pred):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

class eer(tf.keras.metrics.Metric):
    def __init__(self, name='equal_error_rate', **kwargs):
        super(eer, self).__init__(name=name, **kwargs)
        self.score = self.add_weight(name='eer', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.score.assign_add(tf.reduce_sum(tf.py_function(func=compute_eer, inp=[y_true, y_pred], Tout=tf.float32,  name='compute_eer')))
        self.count.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.score, self.count)