import sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Dual_Contrastive_Model, Recommendation_Model
from utils import DataInput1, DataInput2, evaluate

#Note: this code must be run using tensorflow 1.4.0
tf.reset_default_graph()

random.seed(625)
np.random.seed(625)
tf.set_random_seed(625)
batch_size = 32
hidden_size = 128
epoch = 1
lr = 1
K = 6
D = 6

### Static Feature Representation Stage ###
data = pd.read_csv('youku.txt', low_memory=False)
data = data[:len(data)//batch_size*batch_size]
user_count = max(list(data['user_id']))+2
item_count = max(list(data['item_id']))+2
data_set = data.values.tolist()
shuffle_set = data.values.tolist()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Dual_Contrastive_Model(user_count, item_count, hidden_size, batch_size, K, D)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    sys.stdout.flush()
    start_time = time.time()
    
    for _ in range(epoch):
        random.shuffle(shuffle_set)
        loss_sum = 0.0
        for _, uij in DataInput1(shuffle_set, batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss
        print('Epoch %d Train_Loss: %.4f Cost time: %.2f' % (model.global_epoch_step.eval(), loss_sum, time.time()-start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
    #model.save(sess)

    collect_user_code, collect_item_code = [], []
    for _, uij in DataInput1(data_set, batch_size):
        user_code, item_code = model.generate_code(sess, uij)
        collect_user_code.append(user_code)
        collect_item_code.append(item_code)
collect_user_code = [list(x) for y in collect_user_code for x in y]
collect_item_code = [list(x) for y in collect_item_code for x in y]
data['user_code'] = collect_user_code
data['item_code'] = collect_item_code
data.to_csv('youku_with_code.txt',index=False)

### Recommendation Stage (Static Feature + Dynamic Feature) ###
data = pd.read_csv('youku_with_code.txt', low_memory=False)
data['user_code'] = data['user_code'].map(eval)
data['item_code'] = data['item_code'].map(eval)
user_count = max(list(data['user_id']))+2
item_count = max(list(data['item_id']))+2

data = data.sample(frac=1)
validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]
train_data = train_data[:len(train_data)//batch_size*batch_size]
test_data = test_data[:len(test_data)//batch_size*batch_size]
train_set = train_data.values.tolist()
test_set = test_data.values.tolist()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Recommendation_Model(user_count, item_count, hidden_size, batch_size, K, D)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print('AUC: %.4f\t Hit Rate: %.4f' % evaluate(sess, model, test_set))
    sys.stdout.flush()
    lr = 1
    start_time = time.time()
    last_auc = 0.0
    
    for _ in range(epoch):
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / batch_size)
        loss_sum = 0.0
        for _, uij in DataInput2(train_set, batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss
        print('Epoch %d Train_Loss: %.4f' % (model.global_epoch_step.eval(), loss_sum))      
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        print('AUC: %.4f\t Hit Rate: %.4f' % evaluate(sess, model, test_set))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()