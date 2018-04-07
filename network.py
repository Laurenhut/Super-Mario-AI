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


tf.logging.set_verbosity(tf.logging.INFO)
m = PyMouse()
k = PyKeyboard()

steps=20000
episodes=1000
batchsize=32
replaysize=500000
epsilon= 0.1
es=0.1
ef= 0.0001
tot=50000

actions=['x',k.right_key ]

# actions=['nothing','z','x','a','jright','jleft','sleft','sright',k.up_key,k.down_key,k.left_key,k.right_key]
numact=len(actions)
action_replay=deque([], maxlen=replaysize)

def screengrab(th,ystanding, oldscore,xold):
    with mss.mss() as sct:

        # Part of the screen to capture
        monitor = {'top': 50, 'left': 65, 'width': 510, 'height': 450}
        # output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)

        # while 'Screen capturing': ENABLE for video feed

        # # Get raw pixels from the screen, save it to a Numpy array
        # mspic=sct.grab(monitor)
        # img = np.array(mspic)
        # imgorig = np.array(mspic)

        stuff=[0]*11
        stuff1=[0]*11
        stuff2=[0]*11
        stuff3=[0]*11
        stuff4=[0]*11
        stuff5=[0]*11
        x=[0]*1
        y=[0]*1
        score=[0]*1
        gnd=[0]*1
        end=[0]*1
        counter=[0]*1

        k.tap_key('3')
        #collects the data for the new state
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        # Get raw pixels from the screen, save it to a Numpy array
        mspic=sct.grab(monitor)
        img = np.array(mspic)
        imgorig = np.array(mspic)
        #  Wait for next request from client
        x[0] = socket.recv()
        y[0] = socket.recv()
        end[0] = socket.recv()
        score[0] = socket.recv()
        gnd[0] = socket.recv()

        for i in range(len(stuff)):
            stuff[i]=socket.recv()
            stuff1[i]=socket.recv()
            stuff2[i]=socket.recv()
            stuff3[i]=socket.recv()
            stuff4[i]=socket.recv()
            stuff5[i]=socket.recv()
        counter[0] = socket.recv()

        print("Received x: %s " % x[0])
        print("Received y: %s " % y[0])
        print("Received end: %s " % end[0])
        print("Received score: %s " % score[0])
        print("Received gnd: %s " % gnd[0])

        for i in range(len(stuff)):

            print("Received id: %s " % stuff[i])
            print("Received contact?: %s " % stuff1[i])
            print("Received x: %s " % stuff2[i])
            print("Received x speed: %s " % stuff3[i])
            print("Received y: %s " % stuff4[i])
            print("Received y speed: %s " % stuff5[i])
        print("Received counter: %s " % counter[0])

        st=map(int,x)+map(int,y)+map(float,end)+map(int,score)+map(int,gnd)+map(int,stuff)+map(int,stuff1)+map(int,stuff2)+map(float,stuff3)+map(int,stuff4)+map(float,stuff5)+map(int,counter)
        print(st)
        # print(len(st))

        k.tap_key('3')
        #st= all ofthis in array form

        #gets the value of mario's standing height and score
        if th==0:

            yarr=map(int,y)
            scorearr=map(int,score)
            xarr=map(int,x)
            ystanding=yarr[0]
            xold=xarr[0]
            oldscore=scorearr[0]

        #mapping the strings comung rom emulator to integers
        y=map(int,y)
        score=map(int,score)
        x=map(int,x)
        end=map(float,end)


        if end[0]>0.0: #hidden zeros? x 4959
            print ("goal!")
            reward=100
            atend=True
            xold=x[0]

        elif y[0] > ystanding+25 :
            print ("dead")
            reward=-100
            atend=True
            xold=x[0]

        # elif score[0]> oldscore:
        #     oldscore=score[0]
        #     print ("score inc")
        #     reward=10
        #     atend=False
        #     xold=x[0]
        elif x[0]>xold+5:
            reward= 1
            print("right")
            xold=x[0]
            atend=False
        else:
            print ("nothing happening")
            reward=0
            atend=False
            xold=x[0]


        th+=1
        # prints out some info
        print (" x, y, ystanding, oldscore, score, end, gnd")
        print (x[0],y[0], ystanding, oldscore, score[0], end[0], gnd[0])
        print ("  ")

        return imgorig,reward,atend,th,ystanding, oldscore,xold


def keypress(index,th,ystanding, oldscore,xold,oldinp):

    print ("index")
    print(index)
    press=actions[index]
    print ("press val")

    # # print(press)
    # #special combine commands
    # if (press=='jright'):
    #     #release all keys
    #     k.release_key('x')
    #     k.release_key('z')
    #     k.release_key(k.right_key)
    #     k.release_key(k.up_key)
    #     k.release_key(k.down_key)
    #     k.release_key(k.left_key)
    #
    #     print('jright')
    #     # k.tap_key('3')
    #     k.press_key(k.right_key)
    #     k.press_key('z')
    #
    #     #collects the data for the new state
    #     st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)
    #
    #
    #
    #
    # elif (press=='sright'):
    #     #release all keys
    #     k.release_key('x')
    #     k.release_key('z')
    #     k.release_key(k.right_key)
    #     k.release_key(k.up_key)
    #     k.release_key(k.down_key)
    #     k.release_key(k.left_key)
    #
    #     #press the next key
    #     # k.tap_key('3')
    #     print('sright')
    #     k.press_key(k.right_key)
    #     k.press_key('x')
    #
    #
    #     #collects the data for the new state
    #     st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)


    if (press=='nothing'):
        #release all keys
        k.release_key('x')
        k.release_key('z')
        k.release_key(k.right_key)
        k.release_key(k.up_key)
        k.release_key(k.down_key)
        k.release_key(k.left_key)

        # k.tap_key('3')
        print (" nothing")
        oldinp=press


        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore,xold=screengrab(th,ystanding, oldscore,xold)


    #normal cases
    else:
        #release all keys
        k.release_key('x')
        k.release_key('z')
        k.release_key(k.right_key)
        k.release_key(k.up_key)
        k.release_key(k.down_key)
        k.release_key(k.left_key)

        #press the next key
        if (press==k.up_key):
            print('up')
            k.press_key(press)
            oldinp=press

        elif(press==k.down_key):
            print('down')
            k.press_key(press)
            oldinp=press
        elif(press==k.left_key):
            print('left')
            k.press_key(press)
            oldinp=press
        elif (press==k.right_key):
            print('right')
            k.press_key(press)
            oldinp=press
        else:
            if (oldinp==press):
                print (press+"nothing")
                # print (" nothing")
                oldinp=7
                k.release_key('x')
                k.release_key('z')
            else:
                print (press)
                k.press_key(press)
                oldinp=press
            # k.press_key(press)
            # k.tap_key(press)
            # k = PyKeyboard()






        # k.tap_key('3')
        # k.press_key(press)

        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore,xold=screengrab(th,ystanding, oldscore,xold)
        k.release_key('x')
        k.release_key('z')

    return st,reward,atend,th,ystanding, oldscore,xold,oldinp

def cnn( ):
    #state, outputs
    # Our application logic will be added here

    # Input Layer state["x"]
    # input_layer = tf.reshape(state, [None, 88, 88, 4])

    input_layer = tf.placeholder("float",shape= [None, 88, 88, 4])

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
      padding="SAME",
      activation=tf.nn.relu,
      use_bias=True
      )

    # conv1=tf.nn.relu(tf.nn.conv2d(input=input_layer,
    # filter=32,
    # strides=[1,4,4,1],
    # padding="SAME"))
    # # Pooling Layer #1
    p=tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=p,
      filters=64,
      kernel_size=[4, 4],
      strides=(2,2),
      padding="SAME",
      activation=tf.nn.relu,
      use_bias=True)

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
        padding="SAME",
        activation=tf.nn.relu,
        use_bias=True)

    # conv3=tf.nn.relu(tf.nn.conv2d(input=conv2,
    # filter=32,
    # strides=[1,1,1,1],
    # padding="SAME"))
    #flattens the image 11 * 11 * 64
    conv3_flat = tf.reshape(conv3, [-1, 2304])
    #width of the  fature maps may be wrong

    # hidden fully conneced layer
    dense = tf.layers.dense(inputs=conv3_flat, units=512, activation=tf.nn.relu)
    # performs classification of the frature found by the convolutional layers

    # output layer fully connected
    out= tf.layers.dense(inputs=dense, units= numact) #tf.nn.softmax
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
# def cnn2( ):
#
#     #input x, xspeed, y, yspeed,onground, spinjump,blocked?,isducking,direction,coin?,collision, atend, (isenemyactive,enenmyx,enemyxspeed, enemyy,enemyyspeed)*10(12)
#     #input size is [57,1] or [62,1 []
#     input_layer = tf.placeholder("float",shape= [ None, 72])
#
#     # The number of hidden neurons should be between the size of the input layer and the size of the output layer.
#     # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
#     # The number of hidden neurons should be less than twice the size of the input layer.
#     conv3_flat = tf.reshape(input_layer, [-1, 72])
#     #number of neurons should be 1.5-3x that of my input
#     # 1st layer
#     dense1 = tf.layers.dense(inputs=conv3_flat, units=100, activation=tf.nn.relu)
#
#
#
#     # hidden fully conneced layer
#     # dense2 = tf.layers.dense(inputs=dense1, units=150, activation=tf.nn.relu)
#
#     # dense3 = tf.layers.dense(inputs=dense2, units=150, activation=tf.nn.relu)
#     # dense4 = tf.layers.dense(inputs=dense3, units=150, activation=tf.nn.softmax)
#     # performs classification of the frature found by the convolutional layers
#
#     # output layer fully connected
#     out= tf.layers.dense(inputs=dense1, units= numact) #tf.nn.softmax
#     # returns the raw outputs for each of the possible button values
#
#
#     return out, input_layer
def actionreplay(state,action,reward,state2,isend):
    action_replay.append((state,action,reward,state2,isend))


def Qlearn(episode, steps, prob,sess):
    gamma= 0.99
    learnrate=.618
    k = PyKeyboard()

    #runs the cnn model to create initial values of all my values
    states, input_layer=cnn( )

    act = tf.placeholder("float",[None, numact])
    nextQ=tf.placeholder("float",[None])

    #for performing gradient decent and minimizing loss
    stateact = tf.reduce_sum(tf.multiply(states, act), reduction_indices=1)
    loss= tf.reduce_mean(tf.square(nextQ - stateact))
    trainer = tf.train.AdamOptimizer(1e-6)
    updateModel =  trainer.minimize(loss)


    #saves training data
    saver=tf.train.Saver()
    # if i==0:
    st=tf.train.get_checkpoint_state('./')
    if st and st.model_checkpoint_path:
        saver.restore(sess,st.model_checkpoint_path)
        print ("loaded")
    else:
        print ("nothing to restore")
    #initializesall my variables so i can now print them after they have been evaluated
    sess.run(tf.global_variables_initializer())
    count=0


    #LOOP FOR THE NUMBER OF EPISODES I WANT TO USE
    for i in range(episode):
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
        epsilon= 0.1
        es=0.1
        ef= 0.0001
        tot=50000



        #make first action [right]
        index=1
        oldinp=3
        #will call the next state and return new state info, rewards,and misc
        img,reward,atend,th,ystanding, oldscore,xold,oldinp=keypress(index,th,ystanding, oldscore,xold,oldinp)
        img=cv2.cvtColor( cv2.resize(img,(88,88)), cv2.COLOR_BGR2GRAY)
        s_t=np.stack((img, img, img, img), axis=2)


        #iterates through number of steps in each episode
        for i in range(steps):

            # print(loss)
            # out=tf.Print(loss,[loss],'hi')
            #will put all my Q values into an array[][]
            readout_t = states.eval(feed_dict={input_layer : [s_t]})
            print("readout")
            print (readout_t) #<- will print out the cnn state values


            # will either ge max q or a random q based on episilon e
            if random.random()<epsilon:
                print("random")
                #chooses a random action (takes the index)
                a1=random.choice(list(enumerate(readout_t[0])))[0]

            else:
                print("max")
                #gets the max Q value
                a1=np.argmax(readout_t[0])

            print (a1)

            if epsilon > ef and count > 100:
                epsilon -= (es- ef) / tot

            # #makes an action and returnst state 2, reward
            img2,reward,atend,th,ystanding, oldscore,xold,oldinp=keypress(a1,th,ystanding, oldscore,xold,oldinp)
            # cv2.imwrite('st2.png',s_t2)
            #gets total reward
            img2=cv2.cvtColor( cv2.resize(img2,(88,88)), cv2.COLOR_BGR2GRAY)
            # ret, img2 = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)
            img2 = np.reshape(img2, (88, 88, 1))

            totreward=totreward+reward
            s_t2=np.append(img2, s_t[:, :, :3], axis=2)
            print("total reward")
            print(totreward)
            z=np.zeros([numact])
            a_t = np.zeros([numact])

            z[a1]=1

            #store transition from S1->s2  saved in replay memory
            actionreplay(s_t,z,reward,s_t2,atend)
            if len(action_replay)==replaysize:
                action_replay.popleft()

            #after count # of frames will begin training on a batch of previous states
            if count>=100:
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


                #updates the model with the new Q values
                tt=updateModel.run(feed_dict={ nextQ: Q,act:abat, input_layer : sta2})
                # tt=updateModel.run(feed_dict={act:abat, nextQ: Q, input_layer : sta1})
                print("updated")
                print(totreward)


            count+=1
            #saves the model after 1k steps
            if i%1000==0:
                saver.save(sess, './mario.ckpt', global_step=i, write_state=True )

            #if we have died or reached the end terminate episode else continue
            if atend ==True:

                print("restart")
                break
            else:
                #set the new state
                s_t =s_t2


            #end of for
        #end of for

        k = PyKeyboard()
        print("restart")
        #resets the stage
        k.tap_key('q')

def main():
    sess = tf.InteractiveSession()
    i=0
    while i<5:
        print(i)
        time.sleep(1)
        i+=1

    k.tap_key('w')
    k.tap_key('1')
    # Qlearn(1,60,.99,sess)
    Qlearn(episodes,steps,.99,sess)


if __name__ == "__main__":

  main()
