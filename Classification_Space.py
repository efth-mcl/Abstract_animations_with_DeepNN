import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import os
################
def ColorClass(D_S,Hot):
    D = D_S.shape[1]
    K = Hot.shape[1]
    colors = np.random.rand(K,3)
    C = np.zeros((D_S.shape[0],3))
    for i in range(K):
        C[np.where(np.argmax(Hot,axis = 1)==i)]=colors[i]
    return C

def readPointsSet():
    examples_per_class=100
    n_class=4
    T_Examples=examples_per_class*n_class
    t = np.linspace(np.pi/4,np.pi*2,examples_per_class)
    thita = np.random.rand(1)*2*np.pi + np.arange(n_class)*2*np.pi/n_class
    thita=thita.reshape(n_class,1)
    xt = t/2*np.pi*np.cos(t+thita)+np.random.rand(t.size)
    yt = t/2*np.pi*np.sin(t+thita)+np.random.rand(t.size)
    D_S = np.array([xt.reshape(xt.size),yt.reshape(yt.size)]).T
    Hot = np.zeros((T_Examples,n_class))
    for i in range(n_class):
        Hot[examples_per_class*i : examples_per_class*(i+1) ,i]=1
    Rarray = np.random.choice(np.arange(0, T_Examples), replace = False, size=(1, T_Examples)).reshape(T_Examples)
    D_S = D_S[Rarray]
    Hot = Hot[Rarray]
    return D_S,Hot,ColorClass(D_S,Hot)


X,Y,C = readPointsSet()

res = 400
Min=min(min(X[:,0]),min(X[:,1]))
Max=max(max(X[:,0]),max(X[:,1]))
xi = Min; xf =Max; numx = res
yi = Min; yf = Max; numy = res
px = xi
s=(xf-xi)/numx
S_c=[] # All points in the rectangle area with corners (xi, yi) , (xf, yf)
for i in range(numx):
    px+=s
    py = yi
    for j in range(numy):
        py+=s
        S_c.append([px,py])
S_c = np.array(S_c).reshape(len(S_c),2)

D = X.shape[1]  # number of input vector dimensions
K = Y.shape[1]  # number of classes

plt.title(str(K)+' Spirals')
plt.scatter(X[:, 0], X[:, 1], c = C, s = 40, cmap = plt.cm.Spectral)
plt.show()



Input = tf.placeholder(tf.float32, [None, D])
TrOut = tf.placeholder(tf.float32, [None, K]) #True Output

#1st Hiden Full Donnected Layer
W_fc1 = tf.Variable(tf.random_uniform([D,25]))*0.1
b_fc1 = tf.Variable(tf.random_uniform([25]))*0.1
h_fc1 = tf.nn.relu(tf.matmul(Input, W_fc1) + b_fc1)

#2st hiden FC
W_fc2 = tf.Variable(tf.random_uniform([25,50]))*0.1
b_fc2 = tf.Variable(tf.random_uniform([50]))*0.1
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#3st hiden FC
W_fc3 = tf.Variable(tf.random_uniform([50,100]))*0.1
b_fc3 = tf.Variable(tf.random_uniform([100]))*0.1
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

#output FC
W_fc4 = tf.Variable(tf.random_uniform([100,K]))*0.1
b_fc4 = tf.Variable(tf.random_uniform([K]))*0.1
PrOut = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

#Cross Entropy
loss = tf.reduce_mean(-tf.reduce_sum(TrOut * tf.log(PrOut+1e-12), reduction_indices=[1]))

#Regularization for smoother aproximation
reg = tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc3)+tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_fc4)
loss = tf.reduce_mean(loss+0.001*reg)

#Accuracy
correct_prediction = tf.equal(tf.argmax(PrOut,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


fig,ax = plt.subplots()
Z=[]
Loss = 100
i = 0
while Loss>6e-2:
    train_step.run(feed_dict={Input: X, TrOut: Y})
    if i%10==0:
        print('step',i)
        Loss = loss.eval(feed_dict={Input: X, TrOut: Y})
        print('LOSS',Loss)
        print('ACURACY',accuracy.eval(feed_dict={Input: X, TrOut: Y}))
        z = PrOut.eval( feed_dict={Input: S_c})
        z = (np.argmax(z, axis = 1)).reshape(numx,numy)
        Z.append(np.rot90(z, 1))
    i+=1

def animate(i):
    ax.clear()
    extent = (xi, xf, yi, yf)
    ax.imshow(Z[i], extent = extent)

ani = animation.FuncAnimation(fig, animate, frames = len(Z), interval = 10, blit= False, repeat = True)

plt.show()
