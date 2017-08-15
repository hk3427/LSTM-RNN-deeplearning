
# coding: utf-8

# In[1]:

import math
import pickle as p
import tensorflow as tf
import numpy as np
import utils
import json


# In[2]:

tweet_size = 20
hidden_size = 100
vocab_size = 7597
batch_size = 64

tf.reset_default_graph()
session = tf.Session()


# In[3]:

#Placeholder for variable data
tweets = tf.placeholder(tf.float32, [None, tweet_size, vocab_size])
labels = tf.placeholder(tf.float32,[None])


# In[4]:

#Creating a LSTM Cell
lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)

multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2] , state_is_tuple=True)

_, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32)


# In[5]:

#Computation graph
def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0/ math.sqrt(shape[-1])))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b


# In[6]:

sentiment = linear(final_state[-1][-1], 1, name="output")


# In[7]:

sentiment = tf.squeeze(sentiment, [1])

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sentiment, labels=labels)

loss = tf.reduce_mean(loss)


prediction = tf.to_float(tf.greater_equal(sentiment, 0.5))

# error based on which predictions were actually correct.
pred_err = tf.to_float(tf.not_equal(prediction, labels))
pred_err = tf.reduce_sum(pred_err)


# In[8]:

#Adam Optimizer
optimizer = optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[ ]:


tf.global_variables_initializer().run(session=session)

# Training data
train_data = json.load(open('data/trainTweets_preprocessed.json', 'r'))
train_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),train_data))
train_tweets = np.array([t[0] for t in train_data])
train_labels = np.array([int(t[1]) for t in train_data])
#Test Data
test_data = json.load(open('data/testTweets_preprocessed.json', 'r'))
test_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),test_data))
test_data = test_data[0:1000] 
test_tweets = np.array([t[0] for t in test_data])
one_hot_test_tweets = utils.one_hot(test_tweets, vocab_size)
test_labels = np.array([int(t[1]) for t in test_data])

num_steps = 1000

for step in range(num_steps):
    offset = (step * batch_size) % (len(train_data) - batch_size)
    batch_tweets = utils.one_hot(train_tweets[offset : (offset + batch_size)], vocab_size)
    batch_labels = train_labels[offset : (offset + batch_size)]
    data = {tweets: batch_tweets, labels: batch_labels}
    
    _, loss_value_train, error_value_train = session.run(
      [optimizer, loss, pred_err], feed_dict=data)
    
    
    if (step % 50 == 0):
        print("Minibatch train loss at step", step, ":", loss_value_train)
        print("Minibatch train error: %.3f%%" % error_value_train)
        
        #  test evaluation
        test_loss = []
        test_error = []
        for batch_num in range(int(len(test_data)/batch_size)):
            test_offset = (batch_num * batch_size) % (len(test_data) - batch_size)
            test_batch_tweets = one_hot_test_tweets[test_offset : (test_offset + batch_size)]
            test_batch_labels = test_labels[test_offset : (test_offset + batch_size)]
            data_testing = {tweets: test_batch_tweets, labels: test_batch_labels}
            loss_value_test, error_value_test = session.run([loss, pred_err], feed_dict=data_testing)
            test_loss.append(loss_value_test)
            test_error.append(error_value_test)
        
        print("Test loss: %.3f" % np.mean(test_loss))
        print("Test error: %.3f%%" % np.mean(test_error))


# In[ ]:



