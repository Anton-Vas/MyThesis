#   черновик  ----------------
 
# Load modules
import tensorflow.compat.v1 as tf     #изменил на это, чтоб блокнот импортировал функцию (placeholder)
tf.disable_v2_behavior()
 
import numpy as np
import functools
import tensorflow_probability as tfp
import math, random
import matplotlib.pyplot as plt
from scipy import special
from keras.models import Sequential
 
######################################################################
# Routine to solve u''(x) - x*u(x) = f(x), u(0)=A, u'(0)=B in the form
#     u(x) = A + B*x + x^2*N(x,w)
# where N(x,w) is the output of the neural network.
######################################################################
 
# Create the arrays x and y, where x is a discretization of the domain (a,b) and y is the source term f(x)
N = 200
a = 0#-6.0
b = 1#2.0
x = np.arange(a, b, (b-a)/N).reshape((N,1))
y = np.zeros(N)
 
# Boundary conditions
A = 0.0
B = 0.0

alpha = 180
beta = 25
 
# Define the number of neurons in each layer
n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_hl4 = 200
n_nodes_hl5 = 200
 
# Define the number of outputs and the learning rate
n_classes = 1
learn_rate = 0.00003
 
# Define input / output placeholders
x_ph = tf.placeholder('float', [None, 1],name='input')
y_ph = tf.placeholder('float')
 
# Define standard deviation for the weights and biases
hl_sigma = 0.01 #0.05
 
# Routine to compute the neural network (5 hidden layers)
def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(name='w_h1',initial_value=tf.random_normal([1, n_nodes_hl1], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h1',initial_value=tf.random_normal([n_nodes_hl1], stddev=hl_sigma))}
 
    hidden_2_layer = {'weights': tf.Variable(name='w_h2',initial_value=tf.random_normal([n_nodes_hl1, n_nodes_hl2], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h2',initial_value=tf.random_normal([n_nodes_hl2], stddev=hl_sigma))}
 
    hidden_3_layer = {'weights': tf.Variable(name='w_h3',initial_value=tf.random_normal([n_nodes_hl2, n_nodes_hl3], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h3',initial_value=tf.random_normal([n_nodes_hl3], stddev=hl_sigma))}
 
    hidden_4_layer = {'weights': tf.Variable(name='w_h4',initial_value=tf.random_normal([n_nodes_hl3, n_nodes_hl4], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h4',initial_value=tf.random_normal([n_nodes_hl4], stddev=hl_sigma))}
 
    hidden_5_layer = {'weights': tf.Variable(name='w_h5',initial_value=tf.random_normal([n_nodes_hl4, n_nodes_hl5], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h5',initial_value=tf.random_normal([n_nodes_hl5], stddev=hl_sigma))}
 
    output_layer = {'weights': tf.Variable(name='w_o',initial_value=tf.random_normal([n_nodes_hl5, n_classes], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_o',initial_value=tf.random_normal([n_classes], stddev=hl_sigma))}
 
 
    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)   #leaky
 
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = tf.nn.relu6(l2)
 
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    #l3 = tf.nn.relu6(l3) #tanh
 
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    #l4 = tf.nn.relu6(l4) #tanh
 
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    #l5 = tf.nn.relu6(l5) #relu
 
    output = tf.add(tf.matmul(l5, output_layer['weights']), output_layer['biases'], name='output')
    output = tf.nn.swish(output) #swish

    return output
 
batch_size = 32
 
# Feed batch data
def get_batch(inputX, inputY, batch_size):
    duration = len(inputX)
    for i in range(0,duration//batch_size):
        idx = i*batch_size + np.random.randint(0,10,(1))[0]
 
        yield inputX[idx:idx+batch_size], inputY[idx:idx+batch_size]
 
def _make_val_and_grad_fn(value_fn):
    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)
    return val_and_grad

@_make_val_and_grad_fn
def quadratic(x_ph):
    prediction = neural_network_model(x_ph)
    pred_dx = tf.gradients(prediction, x_ph)
    pred_dx2 = tf.gradients(tf.gradients(prediction, x_ph),x_ph)
 
    # Compute u and its second derivative
    u = (x_ph - 1)*(x_ph - 1)*x_ph*x_ph*prediction
    
    dudx2 = (x_ph*x_ph)*pred_dx2 + 2.0*x_ph*pred_dx + 2.0*x_ph*pred_dx + 2.0*prediction

    return tf.reduce_mean(tf.square(dudx2-x_ph*u - y_ph))
 
# Routine to train the neural network
def train_neural_network_batch(x_ph, predict=False):
    prediction = neural_network_model(x_ph)
    pred_dx = tf.gradients(prediction, x_ph)
    pred_dx2 = tf.gradients(tf.gradients(prediction, x_ph),x_ph)
    pred_dx3 = tf.gradients(tf.gradients(tf.gradients(prediction, x_ph),x_ph),x_ph)
    pred_dx4 = tf.gradients(tf.gradients(tf.gradients(tf.gradients(prediction, x_ph),x_ph),x_ph),x_ph)

    # Compute u and its second derivative
    #u = A + B*x_ph + (x_ph*x_ph)*prediction
    #dudx2 = (x_ph*x_ph)*pred_dx2 + 2.0*x_ph*pred_dx + 2.0*x_ph*pred_dx + 2.0*prediction
    u = (x_ph*x_ph - 1)*(x_ph*x_ph - 1)*x_ph*x_ph*prediction
    dudx4 = 24.0*prediction + 48.*(2*x_ph -1)*pred_dx + 12.*(6*x_ph*x_ph - 3*x_ph + 1)*pred_dx2 + 8.*(2*x_ph*x_ph*x_ph - 3*x_ph*x_ph + x_ph)*pred_dx3 + 4.*(x_ph*x_ph - 2*x_ph*x_ph*x_ph + x_ph*x_ph*x_ph*x_ph)*pred_dx4

    # The cost function is just the residual of u''(x) - x*u(x) = 0, i.e. residual = u''(x)-x*u(x)
    cost = tf.reduce_mean(tf.square(dudx4-alpha*u - beta - y_ph))
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    
    #start = tf.constant([-1.2, 1.7])
    #optimizer = tfp.optimizer.lbfgs_minimize(quadratic, initial_position=start, tolerance=1e-5) 
 
    # cycles feed forward + backprop
    hm_epochs = 100
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        # Train in each epoch with the whole data
        for epoch in range(hm_epochs):
 
            epoch_loss = 0
            for step in range(N//batch_size):
                for inputX, inputY in get_batch(x, y, batch_size):
                    _, l = sess.run([optimizer,cost], feed_dict={x_ph:inputX, y_ph:inputY})
                    epoch_loss += l
            if epoch %10 == 0:
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
 
 
        # Predict a new input by adding a random number, to check whether the network has actually learned
        x_valid = x + np.random.normal(scale=0.1,size=(1))
        return sess.run(tf.squeeze(prediction),{x_ph:x_valid}), x_valid
 
 
# Train network
tf.set_random_seed(42)
pred, time = train_neural_network_batch(x_ph)
 
 
mypred = pred.reshape(N,1)
 
# Compute Airy functions for exact solution
ai, aip, bi, bip = special.airy(time)
 
# Numerical solution vs. exact solution
fig = plt.figure()
plt.plot(time, time*time*(time -1)*(time -1)*mypred)
plt.xlim(0, 1)
plt.ylim(0.00, 0.1)
#plt.plot(time, 0.5*(3.0**(1/6))*special.gamma(2/3)*(3**(1/2)*ai + bi))
#plt.plot(time, (-3.0**(1/3)*special.gamma(1/3)-3.0**(2/3)*special.gamma(2/3))*ai+(3.0**(-1/6)*special.gamma(1/3)-3.0**(7/6)*special.gamma(2/3))*bi)
plt.show()