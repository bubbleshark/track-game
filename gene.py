import tensorflow as tf
from itertools import chain
import numpy as np
import math
import random
import os
import copy
from mem_top import mem_top
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
def mate_weights(trainer,sort_list,player_num,select_rate,max_weights):
    select_pool = []
    new_pool = []
    for i in range(0,math.ceil(player_num*select_rate)):
        l = trainer[sort_list[i][0]].get_weights()
        select_pool.append(l)
    select_num = len(select_pool)
    
    node_count = 0
    for i in range(0,len(select_pool[0])):
        for i2 in range(0,len(select_pool[0][i])):
            for i3 in range(0,len(select_pool[0][i][i2])):
                node_count += 1
    '''
    for i in range(0,player_num):
        l=[]
        for i2 in range(0,len(select_pool[0])): # get layers
            l2 = []
            for i3 in range(0,len(select_pool[0][i2])):
                l3 = []
                for i4 in range(0,len(select_pool[0][i2][i3])):
                    l3.append(select_pool[math.floor(random.random()*select_num)][i2][i3][i4])
                l2.append(l3)
            l.append(l2)
    '''
    for i in range(0,player_num):
        l = copy.deepcopy(max_weights)
        if i == 0:
            new_pool.append(l)
            continue
        for i2 in range(0,math.ceil(node_count*0.1)):
            i3 = random.randint(0,len(l)-1) # select layer
            i4 = random.randint(0,len(l[i3])-1)
            i5 = random.randint(0,len(l[i3][i4])-1)
            l[i3][i4][i5] = random.uniform(-2, 2)
        #if i == 0:
        #    print(l[0][0])
        new_pool.append(l)
    return new_pool

def mate_biases(trainer,sort_list,player_num,select_rate,max_biases):
    select_pool = []
    new_pool = []
    for i in range(0,math.ceil(player_num*select_rate)):
        l = trainer[sort_list[i][0]].get_biases()
        select_pool.append(l)
    select_num = len(select_pool)
    
    node_count = 0
    for i in range(0,len(select_pool[0])):
        for i2 in range(0,len(select_pool[0][i])):
            node_count += 1
    
    for i in range(0,player_num):
        l= copy.deepcopy(max_biases)
        if i == 0:
            new_pool.append(l)
            continue
        for i2 in range(0,math.ceil(node_count*0.1)):
            i3 = random.randint(0,len(l)-1) # select layer
            i4 = random.randint(0,len(l[i3])-1)
            l[i3][i4] = random.uniform(-2, 2)
        #if i == 0:
        #    print(l[0])
        new_pool.append(l)
    return new_pool
        
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
        self.layers = []
        self.update_weights = []
        self.update_weights_op = []
        self.update_biases = []
        self.update_biases_op = []
        self.output=[False,False,False,False]
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
        for i in range(0,self.n_layer_num+1):
            if i == 0:
                self.layers.append(tf.add(tf.matmul(self.X, self.weights[i]),self.biases[i]))
            else:
                self.layers.append(tf.add(tf.matmul(self.layers[i-1], self.weights[i]),self.biases[i]))
        for i in range(0,len(self.weights)):
            self.update_weights.append(tf.placeholder(dtype=tf.float32, shape=self.weights[i].get_shape()))
            self.update_weights_op.append(self.weights[i].assign(self.update_weights[i]))
        for i in range(0,len(self.biases)):
            self.update_biases.append(tf.placeholder(dtype=tf.float32,shape=self.biases[i].get_shape()))
            self.update_biases_op.append(self.biases[i].assign(self.update_biases[i]))
        
    def neural_net(self):
        return self.layers[-1]
        
    def run(self,input):
        tp_output=self.sess.run(self.neural_net(),feed_dict={self.X: input})[0]    
        for i in range(0,4):
            if tp_output[i] >= 0.0:
                self.output[i] = True
            else:
                self.output[i] = False
        return
        
    def get_weights(self):
        l = list(self.sess.run(self.weights))
        for i in range(0,len(l)):
            l[i] = list(l[i])
            for k in range(0,len(l[i])):
                l[i][k] = list(l[i][k])
        return l
    
    def get_biases(self):
        l = list(self.sess.run(self.biases))
        for i in range(0,len(l)):
            l[i] = list(l[i])
        return l
    def set_weights(self,weights):
        for i in range(0,len(weights)):
            self.sess.run(self.update_weights_op[i],feed_dict={self.update_weights[i]:np.array(weights[i])})
        return
        
    def set_biases(self,biases):
        for i in range(0,len(biases)):
            self.sess.run(self.update_biases_op[i],feed_dict={self.update_biases[i]:np.array(biases[i])})
        return
        
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