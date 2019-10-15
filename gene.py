import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
train_info={
    "n_layer_num": 2,
    "n_hidden":[10,10],
    "num_input":10,
    "num_output": 4
}

class Trainer():
    def __init__(self,player_info,train_info):
        self.player_info = player_info
        self.n_layer_num = train_info["n_layer_num"]
        self.n_hidden = train_info["n_hidden"]
        self.num_input = train_info["num_input"]
        self.num_output = train_info["num_output"]
        self.X = tf.placeholder("float", [None, self.num_input])
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
            print(i)
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
        print(output)

def main():
    trainer = Trainer(player_info,train_info)
    input = [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]] # n-1, n
    sensor_num = len(player_info["sensor_offset"])
    for i,e in enumerate(player_info["sensor_offset"]):
        #print(i)
        input[0][i] = input[0][i]/e[2]
        input[0][i+sensor_num] = input[0][i]/e[2]
    trainer.run(input)

main()