from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from collections import deque
import random
import cv2
import time
from pymouse import PyMouse
from pykeyboard import PyKeyboard
import zmq
import mss

import gym

env=gym.make('SpaceInvaders-v0')

tf.logging.set_verbosity(tf.logging.INFO)
m = PyMouse()
k = PyKeyboard()

steps=70000
episodes=1000
batchsize=32
replaysize=500000
epsilon= 0.1
es=0.1
ef= 0.0001
tot=50000
Stuff=False

actions=[0,1,2,3 ]

# actions=['nothing','z','x','a','jright','jleft','sleft','sright',k.up_key,k.down_key,k.left_key,k.right_key]
numact=len(actions)
action_replay=deque([], maxlen=replaysize)




def convert():
    input_state_conv = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
    output = tf.image.rgb_to_grayscale(input_state_conv)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_state_conv = tf.squeeze(output)
    #return input_state_conv

def process(inp):
    #input_state_conv = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
    #input_state_conv =tf.convert_to_tensor(inp, dtype=tf.uint8)
    #output = tf.image.rgb_to_grayscale(input_state_conv )
    #output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    #output = tf.image.resize_images(output, [88, 88], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #input_state_conv = tf.squeeze(output)
    return cv2.cvtColor( cv2.resize(inp,(88,88)), cv2.COLOR_BGR2GRAY)





def cnn( ):
    #state, outputs
    # Our application logic will be added here

    # Input Layer state["x"]
    # input_layer = tf.reshape(state, [None, 88, 88, 4])

    input_layer = tf.placeholder("float",shape= [None, 88, 88, 4])
    #input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)


    #converts the input image into the desired shape
    #-1 = the match size dimension is dynamiclly computed based on the input values in features
    # i want to use [3,84,84,1] - moochrome, 84x84 pixel image feed in 3 at a time
    # batch_size, image_width, image height, channels

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[8, 8],
      strides=(4,4),

      activation=tf.nn.relu

      )

    # conv1=tf.nn.relu(tf.nn.conv2d(input=input_layer,
    # filter=32,
    # strides=[1,4,4,1],
    # padding="SAME"))
    # # Pooling Layer #1
    #p=tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[4, 4],
      strides=(2,2),

      activation=tf.nn.relu
      )

    # conv2=tf.nn.relu(tf.nn.conv2d(input=p,
    # filter=32,
    # strides=[1,2,2,1],
    # padding="SAME"))
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64, #maybe change to 64
        kernel_size=[4, 4],
        strides=(1,1),

        activation=tf.nn.relu
        )

    # conv3=tf.nn.relu(tf.nn.conv2d(input=conv2,
    # filter=32,
    # strides=[1,1,1,1],
    # padding="SAME"))
    #flattens the image 11 * 11 * 64
    #conv3_flat = tf.reshape(conv3, [-1, 2304])
    conv3_flat=tf.contrib.layers.flatten(conv3)
    #width of the  fature maps may be wrong

    # hidden fully conneced layer
    dense = tf.contrib.layers.fully_connected(conv3_flat, 512)
    # performs classification of the frature found by the convolutional layers

    # output layer fully connected
    out= tf.contrib.layers.fully_connected(dense, numact) #tf.nn.softmax
    # returns the raw outputs for each of the possible button values



  # dropout = tf.layers.dropout(
  #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


   # trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
   #                                     scope=scope.name)
   #  trainable_vars_by_name = {var.name[len(scope.name):]: var
   #                            for var in trainable_vars}


    #gets the corresponding names for each outputs

    #optimize it
    return out, input_layer

def copymodel(estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    return update_ops

def actionreplay(state,action,reward,state2,isend):
    action_replay.append((state,action,reward,state2,isend))


def Qlearn(episode, steps, prob,sess,sz):
    gamma= 0.99
    learnrate=.618
    k = PyKeyboard()

    #runs the cnn model to create initial values of all my values
    states, input_layer=cnn( )
    #target=cnn( )
    #qwqe=copymodel(target, target)

    act = tf.placeholder("float",[None, numact])
    nextQ=tf.placeholder("float",[None])

    #for performing gradient decent and minimizing loss
    stateact = tf.reduce_sum(tf.multiply(states, act), reduction_indices=1)
    loss= tf.reduce_mean(tf.square(nextQ - stateact))
    #trainer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    trainer = tf.train.AdamOptimizer(1e-6)
    updateModel =  trainer.minimize(loss)


    #saves training data
    #saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    st=tf.train.get_checkpoint_state('./')
    saver = tf.train.import_meta_graph('Breakout.ckpt-0.meta')
    if st and st.model_checkpoint_path :
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        print ("loaded")
    else:
        print ("nothing to restore")
    #initializesall my variables so i can now print them after they have been evaluated

    count=0

    epsilon= prob
    # epsilon= 0.1
    es=0.1
    ef= 0.0001
    tot=200000
    img=env.reset()
    #LOOP FOR THE NUMBER OF EPISODES I WANT TO USE
    for j in range(episode):
        th=0
        c=0
        xprev=0
        x2prev=0
        x3prev=0
        x4prev=0
        d=0
        e=0
        ystanding=0
        ycurrent=0
        xstanding=0
        y=0
        x=0
        totreward=0
        oldscore=0
        end=0
        gnd=0
        score=0
        xold=0
        '''epsilon= prob
        # epsilon= 0.1
        es=0.1
        ef= 0.0001
        tot=50000'''



        #make first action [right]
        index=0
        oldinp=1
        #k.release_key('x')
        #k.release_key('z')
        #k.release_key(k.right_key)
        #k.release_key(k.up_key)
        #k.release_key(k.down_key)
        #k.release_key(k.left_key)
        #will call the next state and return new state info, rewards,and misc
        #img,reward,atend,th,ystanding, oldscore,xold,oldinp=keypress(index,th,ystanding, oldscore,xold,oldinp)
        #img=cv2.cvtColor( cv2.resize(img,(88,88)), cv2.COLOR_BGR2GRAY)
        #s_t=np.stack((img, img, img, img), axis=2)




        img= process(img)
        #img= state_processor.process(sess, img)
        print(img.shape)
        s_t=np.stack([img]*4, axis=2)
        print(s_t.shape)
        loss=None
        #iterates through number of steps in each episode
        for i in range(steps):

            env.render()
            # print(loss)
            # out=tf.Print(loss,[loss],'hi')
            #will put all my Q values into an array[][]
            readout_t = states.eval(feed_dict={input_layer : [s_t]})
            print("readout")
            print (readout_t) #<- will print out the cnn state values
            print("episode ")
            print(j)
            print("count ")
            print(count)
            print("random ")
            print(epsilon)

            # will either ge max q or a random q based on episilon e
            if random.random()<=epsilon:
                print("random")
                #chooses a random action (takes the index)
                a1=random.choice(list(enumerate(readout_t[0])))[0]
                #a1 = env.action_space.sample()

            else:
                print("max")
                #gets the max Q value
                a1=np.argmax(readout_t[0])

            print("a1 val")
            #print (a1)
            print(actions[a1])

            if epsilon > ef and count > 100 and sz != True:
                epsilon -= (es- ef) / tot

            # #makes an action and returnst state 2, reward
            #img2,reward,atend,th,ystanding, oldscore,xold,oldinp=keypress(a1,th,ystanding, oldscore,xold,oldinp)
            img2,reward,atend,_=env.step(actions[a1])
            img2= process(img2)





            totreward=totreward+reward
            #print (s_t)
            #print(s_t.shape)
            #print(img2.shape)
            #s_t2=np.append(img2, s_t[:, :, :3], axis=2)
            s_t2=np.append( s_t[:, :, 1:],np.expand_dims(img2,2), axis=2)

            print("total reward")
            print(totreward)
            z=np.zeros([numact])

            a_t = np.zeros([1,numact])
            print(actions.index(actions[a1]))
            a_t[0][a1]=1
            print(a1)
            print(a_t[0][0])
            print(actions)
            print(a_t)
            #actions=[1,2 ]
            #print(actions.indexx(actions[a1]))
            #print("a_t")
            #print(actions[a1])
            #print(a_t[0])
            #print(a_t.shape)
            #a_t = np.zeors([1,4])
            #store transition from S1->s2  saved in replay memory

            if len(action_replay)==replaysize:
                action_replay.popleft()
            actionreplay(s_t,a_t[0],reward,s_t2,atend)
            #after count # of frames will begin training on a batch of previous states
            if count>=100and sz !=True:
                print ("acton replay ")
                #sample from minibatch
                D=random.sample(action_replay,batchsize)

                #puts values from the minibatch into arrays
                sta1 = [k[0] for k in D]
                abat = [k[1] for k in D]
                rew = [k[2] for k in D]
                sta2 = [k[3] for k in D]
                term = [k[4] for k in D]

                Q=[]

                #gets q value for s_t2 (second state) for the minibatch
                readout_t2 = states.eval(feed_dict={input_layer : sta2})

                for i in range(len(D)):

                    #if we have died or reached the end  Q=r
                    #otherwuse q= reward+gamma*state2
                    if term[i] ==False:

                        #update the Q value (train it)
                        Q.append(rew[i]+ gamma*np.max(readout_t2[i]))
                    else :
                        Q.append(rew[i])

                bb=np.zeros([1,4])
                #updates the model with the new Q values
                tt=updateModel.run(feed_dict={ nextQ: Q,act:abat, input_layer : sta2})
                #tt=updateModel.run(feed_dict={ nextQ: Q,act:abat, input_layer : sta2})
                # tt=updateModel.run(feed_dict={act:abat, nextQ: Q, input_layer : sta1})
                print("updated")
                print(totreward)


            count+=1
            #saves the model after 1k steps
            if i%1000==0 and sz != True:
                saver.save(sess, './Breakout.ckpt', global_step=i, write_state=True )

            #if we have died or reached the end terminate episode else continue
            if atend ==True:

                print("restart")
                break
            else:
                #set the new state
                s_t =s_t2


            #end of for
        #end of for

        #k = PyKeyboard()
        #print("restart")
        #resets the stage
        #k.tap_key('q')
        img=env.reset()

def main():
    sess = tf.InteractiveSession()
    i=0
    while i<0:
        print(i)
        time.sleep(1)
        i+=1
    if Stuff==True:
        ss=.0001
    else:
        ss=.1
    #k.tap_key('w')
    #k.tap_key('1')
    # Qlearn(1,60,.99,sess)
    Qlearn(episodes,steps,ss,sess,Stuff)


if __name__ == "__main__":

  main()
