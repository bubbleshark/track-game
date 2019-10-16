import tensorflow as tf
from itertools import chain
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''
player_info={
    "rotate_step": 3,
    "initial_x": 121,
    "initial_y": 248,
    "rotate": 270,
    "boost": 0.4,
    "slow_down": 0.1,
    "speed_max": 5,
    "sensor_offset": [[10,0,80,0],[12,340,60,45],[12,20,60,315],[9,316,40,90],[9,44,40,270]] # r,degree,r,degree
}
'''


class Trainer():
    def __init__(self,player_info,train_info):
        self.player_info = player_info
        self.n_layer_num = train_info["n_layer_num"]
        self.n_hidden = train_info["n_hidden"]
        self.num_input = train_info["num_input"]
        self.num_output = train_info["num_output"]
        self.X = tf.placeholder("float", [None, self.num_input])
        self.start = True
        self.score = 0
        self.weights = []
        self.biases = []
        for i in range(0,self.n_layer_num):
            if i == 0:
                self.weights.append(tf.Variable(tf.random_normal([self.num_input, self.n_hidden[i]])))
            else:
                self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[i-1], self.n_hidden[i]])))
        self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[self.n_layer_num-1], self.num_output])))
        for i in range(0,self.n_layer_num):
            self.biases.append(tf.Variable(tf.random_normal([self.n_hidden[i]])))
        self.biases.append(tf.Variable(tf.random_normal([self.num_output])))
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def neural_net(self,x):
        layers = []
        for i in range(0,self.n_layer_num+1):
            if i == 0:
                layers.append(tf.add(tf.matmul(x, self.weights[i]),self.biases[i]))
            else:
                layers.append(tf.add(tf.matmul(layers[i-1], self.weights[i]),self.biases[i]))
        return layers[-1]
        
    def run(self,input):
        tp_output=list(self.sess.run(self.neural_net(self.X),feed_dict={self.X: input}))[0]        
        output = []
        for i in range(0,self.num_output):
            if tp_output[i] >= 0.0:
                output.append(True)
            else:
                output.append(False)
        return output
        
    def get_weights(self):
        l = list(self.sess.run(self.weights))
        for i in range(0,len(l)):
            l[i] = list(l[i])
            for k in range(0,len(l[i])):
                l[i][k] = list(l[i][k])
        #l = np.array(l)
        #l = l.reshape(1)
        #l = [[[1,2,3],[2,2,3],[3,2,3]],[[1,2,4],[2,2,4],[3,2,4]],[[1,2,5],[2,2,5],[3,2,5]]]
        #l = list(chain.from_iterable(list(chain.from_iterable(l))))
        #print(l)
        return l
    
    #def set_weights(self,weights):
    #   [weights[i:i+n] for i in xrange(0, len(weights), n)]

def main():
    trainer = Trainer(player_info,train_info)
    input = [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]] # n-1, n
    sensor_num = len(player_info["sensor_offset"])
    for i,e in enumerate(player_info["sensor_offset"]):
        #print(i)
        input[0][i] = input[0][i]/e[2]
        input[0][i+sensor_num] = input[0][i]/e[2]
    trainer.run(input)

#main()