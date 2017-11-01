import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

dx=0.01
x0=-8
x1=-x0
x=np.linspace(x0, x1, int((x1-x0)/dx))
y=0

for i in range(25):
    c=np.random.randint(2)
    c=(-1)**c
    m=np.random.rand(1)*x1*2*0.9-x1*0.9
    s=np.random.rand(1)*3
    y+=c*np.exp(-(x-m)**2/s)
# y+=np.random.rand(x.size)/6  #add same noise
n_examples=x.shape[0]
Rarray = np.random.choice(np.arange(0, n_examples), replace=False, size=(1, n_examples)).reshape(n_examples)

plt.plot(x,y)
plt.title('This is a function O')
plt.show()
n_examples=200
X=x[Rarray[:n_examples]]
X=X.reshape(X.size,1)
Y=y[Rarray[:n_examples]]
Y=Y.reshape(Y.size,1)

D=X.shape[1]
K=Y.shape[1]


plt.scatter(X,Y)
plt.title('and this is '+str(n_examples)+' random points from O')
plt.xlabel('This points will be used for aproximation of O')
plt.show()

Wrange=0.05
brange=1
Input = tf.placeholder(tf.float32, [None, D])
TrOut = tf.placeholder(tf.float32, [None, K]) #True Output

#1st Hiden Full Donnected Layer
W_fc1 = tf.Variable(tf.random_uniform([D,25],-Wrange,Wrange))
b_fc1 = tf.Variable(tf.random_uniform([25],-brange,brange))
h_fc1 = tf.tanh(tf.matmul(Input, W_fc1) + b_fc1)

#2st hiden FC
W_fc2 = tf.Variable(tf.random_uniform([25,50],-Wrange,Wrange))
b_fc2 = tf.Variable(tf.random_uniform([50],-brange,brange))
h_fc2 = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)

#3st hiden FC
W_fc3 = tf.Variable(tf.random_uniform([50,100],-Wrange,Wrange))
b_fc3 = tf.Variable(tf.random_uniform([100],-brange,brange))
h_fc3 = tf.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

#output FC
W_fc4 = tf.Variable(tf.random_uniform([100,K],-Wrange,Wrange))
b_fc4 = tf.Variable(tf.random_uniform([K],-brange,brange))
PrOut=tf.matmul(h_fc3, W_fc4) + b_fc4

#Mean Square Error
loss1=(TrOut - PrOut)**2
loss2=tf.reduce_sum(loss1,reduction_indices=[1])
loss=tf.reduce_mean(loss2)

#Regularization for smoother aproximation **
# reg=tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc3)+tf.nn.l2_loss(W_fc2)
# loss=tf.reduce_mean(loss+0.001*reg)

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

TestX=np.array(x)
fig,ax = plt.subplots()
Z=[]
Loss=100
i=0
# MinLoss = 6e-2 # with Regularization, smaller MinLoss can't be achieved
MinLoss=1e-3
while Loss>MinLoss:
    train_step.run(feed_dict={Input: X, TrOut: Y})

    if i%50==0:
        print('step',i)
        Loss=loss.eval(feed_dict={Input: X, TrOut: Y})
        print('LOSS',Loss)
        z=PrOut.eval( feed_dict={Input: TestX.reshape(TestX.shape[0],1)})
        Z.append(z.reshape(z.size))
    i+=1


def animate(i):
    ax.clear()
    ax.plot(x,y)
    ax.plot(x,Z[i])

ani = animation.FuncAnimation(fig, animate, frames=len(Z), interval=10, blit= False, repeat = True)
plt.show()
