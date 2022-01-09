import random
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

batch_size = 32

class DataInput1:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        user, item, label = [], [], []
        for record in records:
            user.append(record[0])
            item.append(record[1])
            label.append(record[2])
        return self.i, (user, item, label)

class DataInput2:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        user, item, label, user_code, item_code = [], [], [], [], []
        for record in records:
            user.append(record[0])
            item.append(record[1])
            label.append(record[2])
            user_code.append(record[4])
            item_code.append(record[5])
        return self.i, (user, item, label, user_code, item_code)

def evaluate(sess, model, test_set):
    arr, arr_auc, hit_rate = [], [], []
    userid = list(set([x[0] for x in test_set]))
    for _, uij in DataInput2(test_set, batch_size):
        score, label, user, item = model.test(sess, uij)
        for index in range(len(label)):
            arr.append([score[index], label[index], user[index], item[index]])
            if label[index] >= 0.5:
                arr_auc.append([0, 1, score[index]])
            else:
                arr_auc.append([1, 0, score[index]])

    arr_auc = sorted(arr_auc, key=lambda d:d[2])
    auc, fp1, fp2, tp1, tp2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for record in arr_auc:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_auc) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc = (1.0 - auc / (2.0 * tp2 * fp2))

    for user in userid:
        arr_user = [x for x in arr if x[1] and x[2]==user]
        arr_user = sorted(arr_user, key=lambda d:d[0], reverse = True)[:10]
        arr_user = [x[0] for x in arr_user]
        if len(arr_user) > 0:
            hit_rate.append(sum(arr_user)/len(arr_user))

    return auc, np.mean(hit_rate)