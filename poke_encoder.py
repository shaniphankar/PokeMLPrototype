
# coding: utf-8

# In[83]:

import tensorflow as tf
import numpy as np
import glob
import os
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')


# In[84]:

def plotPoke(x):
    f, a = plt.subplots(2, 8, figsize=(13, 3))
    for i in range(8):
        a[0][i].imshow(x[i], cmap=plt.get_cmap('gray'))
        a[0,i].axis('off')
        a[1][i].imshow(x[i+8], cmap=plt.get_cmap('gray'))
        a[1,i].axis('off')
    f.show()
    plt.draw()


# In[46]:

#import cv2
#sample = np.zeros([40,40,3],dtype='float32')
#print(sample)
#noise_mask = cv2.randn(sample,(0.5),(0.2))
#print(noise_mask[0][:])
#print(sample[0][:])


# In[51]:

#noise_mask = np.random.normal(0.5,0.002,(40,40,3))


# In[85]:

# Create an empty array to store pokemon pics
# this is a list of images, initially it has only no image (size is 40,40,3), we later append the images
orig_img = np.empty((0, 40, 40, 3), dtype='float32')

# Load all images and append into orig_img
path = os.path.abspath("./AE_RGB.ipynb")
path = os.path.dirname(path) + '/'


for pic in glob.glob(path+'Pokemon/*.png'):
    img = mpimg.imread(pic)
    # remove alpha channel  %some alpha=0 but RGB is not equal to [1., 1., 1.]
    
    # img is an array of the pok iimages and first index is width, then height and then RGBA A is the aplha channel.
    img[img[:,:,3]==0] = np.ones((1,4))
    img = img[:,:,0:3]
    
    orig_img = np.append(orig_img, [img], axis=0)

# Use plt to show original images 
plotPoke(orig_img)

noisy_img = np.empty((0, 40, 40, 3), dtype='float32')

for pic in glob.glob(path+'Pokemon/*.png'):
    img = mpimg.imread(pic)
    # remove alpha channel  %some alpha=0 but RGB is not equal to [1., 1., 1.]

    # img is an array of the pok iimages and first index is width, then height and then RGBA A is the aplha channel.
    img[img[:,:,3]==0] = np.ones((1,4))
    img = img[:,:,0:3]
    
    noise_mask = np.random.normal(np.mean(img),0.08,(40,40,3))
    img = noise_mask + img
    
    noisy_img = np.append(noisy_img, [img], axis=0)
    
plotPoke(noisy_img)
  


# In[ ]:

print(orig_img[0][:][:][8])
print(noisy_img[0][:][:][8])


# In[86]:

# Flat all data to one dimension
X_flat = noisy_img.reshape((-1,1600)) 
X_flat_true = orig_img.reshape((-1,1600)) 
    
print ('Original image shape:  {0}\nFlatted image shape:  {1}'.format(orig_img.shape, X_flat.shape))


# In[87]:

# Transpose RGB channels into 3 different independent image
# Then flatted all pixel into one dimension
print(noisy_img.shape)
X_flat = np.transpose(noisy_img, (0,3,1,2))
X_flat_true = np.transpose(orig_img, (0,3,1,2))

print(X_flat.shape)
X_flat = X_flat.reshape(2376, 1600)
X_flat_true = X_flat_true.reshape(2376, 1600)

print('Original image shape:  {0}\nFlatted image shape:  {1}'.format(orig_img.shape, X_flat.shape))


# In[88]:

# Parameters
learning_rate = 0.001
training_epochs = 30000
batch_size = 36
display_step = 2
examples_to_show = 8

# Network Parameters
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 256
n_input = 1600 # Pokemon input (img shape: 40*40)


# In[89]:

# tf Graph input (only pictures)
X_noisy = tf.placeholder("float", [None, n_input])
X_true = tf.placeholder("float", [None, n_input])


weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.01)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.01)),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], stddev=0.01)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.01)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.01))
}
biases = {
    'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.01)),
    'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.01)),
    'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.01)),
    'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.01)),
    'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.01)),
    'decoder_b3': tf.Variable(tf.truncated_normal([n_input], stddev=0.01))
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                  biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                  biases['encoder_b3']))
    return layer_3

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3


# In[90]:

# Construct model
encoder_op = encoder(X_noisy)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X_true

# Define loss and optimizer, minimize the MSE
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# In[91]:

# Initializing the variables
init = tf.global_variables_initializer()

# Create session and graph, initial variables
sess = tf.InteractiveSession()
sess.run(init)


# In[69]:

# Load previous trained model and rewrite to variables, if exists
# Before run this cell, you have to run the cell above first, to define variables and init it.
weightSaver = tf.train.Saver(var_list=weights)
biaseSaver = tf.train.Saver(var_list=biases)

weightSaver.restore(sess, "./saved_model/AE_RGB_weights.ckpt")
biaseSaver.restore(sess, "./saved_model/AE_RGB_biases.ckpt")

print("Model restored.")


# In[92]:

total_batch = int(X_flat.shape[0]/batch_size)
# Training cycle
for epoch in range(3000):
    # Loop over all batches
    start = 0; end = batch_size
    for i in range(total_batch-1):
        index = np.arange(start, end)
        np.random.shuffle(index)
        batch_xs = X_flat[index]
        batch_xs_true = X_flat_true[index]
        start = end; end = start+batch_size
        # Run optimization op (backprop) and loss op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X_noisy: batch_xs, X_true: batch_xs_true})
    # Display logs per epoch step
    if ((epoch == 0) or (epoch+1) % display_step == 0) or ((epoch+1) == training_epochs):
        print('Epoch: {0:05d}   loss: {1:f}'.format(epoch+1, c))

print("Optimization Finished!")


# In[11]:

# Random select some pokemon to visualization
# index are picked in orig_img.shape[0], then transform to X_flat with correspond RGB row
index = np.random.randint(orig_img.shape[0], size=examples_to_show)
index = np.sort(index)
RGB_index = np.concatenate((index*3, index*3+1, index*3+2))
RGB_index = np.sort(RGB_index)
autoencoder = sess.run(
    y_pred, feed_dict={X: X_flat[RGB_index]})

# merge RGB rows back to RGB matrix
autoencoder = np.reshape(autoencoder, (examples_to_show, 3, 40, 40))
autoencoder = np.transpose(autoencoder, (0,2,3,1))

# Compare original images with their reconstructions
f, a = plt.subplots(2, examples_to_show, figsize=(13, 3))
for i in range(examples_to_show):
    a[0][i].imshow(orig_img[index[i]])
    a[0,i].axis('off')
    a[1][i].imshow(autoencoder[i])
    a[1,i].axis('off')
f.show()
plt.draw()


# In[12]:

# Save weights and biases
weights_saver = tf.train.Saver(var_list=weights)
biases_saver = tf.train.Saver(var_list=biases)
weights_saver.save(sess, './saved_model/AE_RGB_weights')
biases_saver.save(sess, './saved_model/AE_RGB_biases')


# In[ ]:



