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


tf.logging.set_verbosity(tf.logging.INFO)
m = PyMouse()
k = PyKeyboard()

steps=20000
episodes=50
batchsize=32
replaysize=500000
epsilon= 0.1

actions=['nothing','z','x','jright','sright',k.up_key,k.down_key,k.right_key ,k.left_key]

# actions=['nothing','z','x','a','jright','jleft','sleft','sright',k.up_key,k.down_key,k.left_key,k.right_key]
numact=len(actions)
action_replay=deque([], maxlen=replaysize)

def screengrab(th,ystanding, oldscore):

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

    st=map(int,x)+map(int,y)+map(int,end)+map(int,score)+map(int,gnd)+map(int,stuff)+map(int,stuff1)+map(int,stuff2)+map(float,stuff3)+map(int,stuff4)+map(float,stuff5)+map(int,counter)
    print(st)
    print(len(st))

    k.tap_key('3')
    #st= all ofthis in array form

    #gets the value of mario's standing height and score
    if th==0:

        yarr=map(int,y)
        scorearr=map(int,score)
        ystanding=yarr[0]
        xstanding=x[0]
        oldscore=scorearr[0]

    y=map(int,y)
    score=map(int,score)


    if 0>end[0]: #hidden zeros?
        print ("goal!")
        reward=100
        atend=True

    elif y[0] > ystanding+50 :
        print ("dead")
        reward=-100
        atend=True

    elif score[0]> oldscore:
        oldscore=score[0]
        print ("score inc")
        reward=1
        atend=False
    else:
        print ("nothing")
        reward=-1
        atend=False


    th+=1
    # prints out some info
    print (" x, y, ystanding, oldscore, score, end, gnd")
    print (x[0],y[0], ystanding, oldscore, score[0], end[0], gnd[0])
    print ("  ")

    return st,reward,atend,th,ystanding, oldscore


def keypress(index,th,ystanding, oldscore):

    print ("index")
    print(index)
    press=actions[index]
    print ("press val")

    # print(press)
    #special combine commands
    if (press=='jright'):
        #release all keys
        k.release_key('x')
        k.release_key('z')
        k.release_key(k.right_key)
        k.release_key(k.up_key)
        k.release_key(k.down_key)
        k.release_key(k.left_key)

        print('jright')
        # k.tap_key('3')
        k.press_key(k.right_key)
        k.press_key('z')

        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)




    elif (press=='sright'):
        #release all keys
        k.release_key('x')
        k.release_key('z')
        k.release_key(k.right_key)
        k.release_key(k.up_key)
        k.release_key(k.down_key)
        k.release_key(k.left_key)

        #press the next key
        # k.tap_key('3')
        print('sright')
        k.press_key(k.right_key)
        k.press_key('x')


        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)


    elif (press=='nothing'):
        #release all keys
        k.release_key('x')
        k.release_key('z')
        k.release_key(k.right_key)
        k.release_key(k.up_key)
        k.release_key(k.down_key)
        k.release_key(k.left_key)

        # k.tap_key('3')
        print (" nothing")


        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)


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

        elif(press==k.down_key):
            print('down')
        elif(press==k.left_key):
            print('left')
        elif (press==k.right_key):
            print('right')
        else:
            print (press)

        # k.tap_key('3')
        k.press_key(press)

        #collects the data for the new state
        st,reward,atend,th,ystanding, oldscore=screengrab(th,ystanding, oldscore)


    return st,reward,atend,th,ystanding, oldscore


def cnn( ):

    #input x, xspeed, y, yspeed,onground, spinjump,blocked?,isducking,direction,coin?,collision, atend, (isenemyactive,enenmyx,enemyxspeed, enemyy,enemyyspeed)*10(12)
    #input size is [57,1] or [62,1 []
    input_layer = tf.placeholder("float",shape= [ None, 72])

    # The number of hidden neurons should be between the size of the input layer and the size of the output layer.
    # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
    # The number of hidden neurons should be less than twice the size of the input layer.

    #number of neurons should be 1.5-3x that of my input
    # 1st layer
    dense1 = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.softmax)



    # hidden fully conneced layer
    dense = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.softmax)
    # performs classification of the frature found by the convolutional layers

    # output layer fully connected
    out= tf.layers.dense(inputs=dense, units= numact, activation=tf.nn.softmax) #tf.nn.softmax
    # returns the raw outputs for each of the possible button values


    return out, input_layer
def actionreplay(state,action,reward,state2,isend):
    action_replay.append((state,action,reward,state2,isend))


def Qlearn(episode, steps, prob,sess):
    gamma= 0.99
    learnrate=.618

    #runs the cnn model to create initial values of all my values
    states, input_layer=cnn( )

    act = tf.placeholder("float",[None, numact])
    nextQ=tf.placeholder("float",[None])

    #for performing gradient decent and minimizing loss
    stateact = tf.reduce_sum(tf.multiply(states, act), reduction_indices=1)
    loss= tf.reduce_mean(tf.square(nextQ - stateact))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)


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



        #make first action [nothing]
        index=0
        #will call the next state and return new state info, rewards,and misc
        st,reward,atend,th,ystanding, oldscore=keypress(index,th,ystanding, oldscore)
        s_t=st

        #iterates through number of steps in the session
        for i in range(steps):


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

            # #makes an action and returnst state 2, reward
            s_t2,reward,atend,th,ystanding, oldscore=keypress(a1,th,ystanding, oldscore)
            # cv2.imwrite('st2.png',s_t2)
            #gets total reward
            totreward=totreward+reward

            z=np.zeros([numact])
            a_t = np.zeros([numact])

            z[a1]=1

            #store transition from S1->s2  saved in replay memory
            actionreplay(s_t,z,reward,s_t2,atend)
            if len(action_replay)==replaysize:
                action_replay.popleft()

            #after count # of frames will begin training on a batch of previous states
            if count>=batchsize:
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
                updateModel.run(feed_dict={act:abat, nextQ: Q, input_layer : sta1})
                print("updated")


            count+=1
            #saves the model after 1k steps
            if i%1000==0:
                saver.save(sess, './mario.ckpt', global_step=i)

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
    while i<10:
        print(i)
        time.sleep(1)
        i+=1

    k.tap_key('w')
    k.tap_key('1')
    # Qlearn(1,60,.99,sess)
    Qlearn(episodes,steps,.99,sess)


if __name__ == "__main__":

  main()
