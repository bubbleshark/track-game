import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
import random
import copy
from track_game_helper import TrackGameHelper
from player import Player
from gene import Trainer, mate_weights, mate_biases
import hashlib
#from mem_top import mem_top

map_file = "data/track4.png"
car_file = "data/car_blue.png"
car_file2 = "data/car_red.png"
obstacle_color = (128,128,128)
white_color = (255, 255, 255, 0)
width, height = 800, 600
fps = 30
user_input = True
player_num = 50
player_info={
    "rotate_step": 3,
    "initial_x": 120,
    "initial_y": 248,
    "rotate": 270,
    "boost": 0.4,
    "slow_down": 0.1,
    "speed_max": 5,
    "sensor_offset": [[10,0,80,0],[12,340,80,45],[12,20,80,315],[9,316,80,90],[9,44,80,270]] # r,degree,r,degree
    #"sensor_offset": [[10,0,80,0],[12,340,60,45],[12,20,60,315],[9,316,40,90],[9,44,40,270]] # r,degree,r,degree
}
train_info={
    "n_layer_num": 2,
    "n_hidden":[10,10],
    "num_input":10,
    "num_output": 4
}
select_rate = 0.2
stop_thres = 3

def print_hash(trainer,player_num):
    score = 0.0
    for k in range(0,player_num):
        tp_w=""
        tp_b=""
        tp_weights = trainer[k].get_weights()
        tp_biases = trainer[k].get_biases()
        _ = [i3 for i in tp_weights for i2 in i for i3 in i2]
        __ = [i2 for i in tp_biases for i2 in i]
        for i in _:
            tp_w = tp_w + str(i)
        for i in __:
            tp_b = tp_b + str(i)
        tp_w = hashlib.md5(tp_w.encode())
        #tp_b = hashlib.md5(tp_b.encode())
        print("trainer "+str(k)+": "+tp_w.hexdigest()+" score: "+str(trainer[k].score))
        score += trainer[k].score
    print("average score:",float(score)/player_num)

def train():
    game = TrackGameHelper(map_file,obstacle_color,white_color,width,height,fps)
    game.set_player(car_file,car_file2,player_info,player_num)
    player_input_list= []
    for i in range(0,player_num):
        player_input_list.append([False,False,False,False])
    trainer = [Trainer(player_info,train_info) for i in range(0,player_num)]
    player_stop = 0
    player_flag = [False for i in range(0,player_num)]
    max_score = -1
    max_weights = None
    max_biases = None
    cou =0
    iter = 0
    #input_tp = [[] for i in range(0,player_num)]
    #output_tp = [[] for i in range(0,player_num)]
    
    while True:
        #print(cou,mem_top())
        cou += 1
        for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    pygame.quit()
                    exit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    for player in game.player:
                        player.reset()
        game.run_step(player_input_list)
        for i,player in enumerate(game.player):
            if trainer[i].start == False:
                continue
            player.start = True
            #print(i)
            #deg = (-1*(math.atan2(player.speed_y, player.speed_x) * (180 / math.pi)) + 360 ) %360
            #print(player.rotate,deg)
            
            forward = False
            #if min(abs(abs(player.rotate - deg)),abs(player.rotate - deg+360)) < 180:
            #    forward = True
            forward = True
            #if abs(player.rotate - deg) > 270 and  pow(player.speed_x,2)+pow(player.speed_y,2) > 0:
            #    forward = False
            
            #for k,e in enumerate(player.sensor_value):
            #    if k == 0:
            #        trainer[i].score += e*4
            #        continue
            #    trainer[i].score += e
            #trainer[i].score += trainer[i].score*math.sqrt(pow(player.speed_x,2)+pow(player.speed_y,2))
            #trainer[i].score += player.sensor_value[0]+(player.sensor_value[-2]+player.sensor_value[-1])/(abs(player.sensor_value[-2]-player.sensor_value[-1])+1)
            #for k in player.sensor_value:
            #    if k <10:
            #        trainer[i].score -= 200
            #trainer[i].score += (pow(player.speed_x,2)+pow(player.speed_y,2)) /(abs(player.sensor_value[-2]-player.sensor_value[-1])+1)
            #if trainer[i].high_score < trainer[i].score:
            #    trainer[i].high_score = trainer[i].score
            #print(player.prev_rotate,player.rotate)
            trainer[i].score += 1 #/(abs(player.prev_rotate-player.rotate)+1)
            #print(trainer[i].score)
            #print(i,player.speed_x,player.speed_y,player.rotate,deg,abs(player.rotate - deg),forward)
            
            #if player.collide == True or (forward == False and player_flag[i] == True) or (player.speed_x==0 and player.speed_y==0 and player_flag[i] == True):
            if player.speed_x==0 and player.speed_y==0:
                trainer[i].stop_cou += 1
            else:
                trainer[i].stop_cou = 0
            if player.collide == True or (forward == False and player_flag[i] == True) or (trainer[i].stop_cou >= stop_thres and player_flag[i] == True):
                player.reset()
                #rint("stop",i)
                if trainer[i].score <= 0.0:
                    trainer[i].score = 0
                    trainer[i].stop_cou = 0
                    trainer[i].high_score=0
                trainer[i].start = False
                player_input_list[i] = [False,False,False,False]
                player_stop += 1
                continue
            #input = [player.sensor_value_prev+player.sensor_value]
            input = [[0,0,0,0,0]+player.sensor_value]
            #input = [[0,0,player.speed_x,player.speed_y,player.rotate]+player.sensor_value]
            
            #exit()
            #input_tp[i].append(input)
            sensor_num = len(player_info["sensor_offset"])
            #print(player.speed_x,player.speed_y)
            trainer[i].run(input)
            all_stop = 0
            for k in range(0,4):
                player_input_list[i][k] = trainer[i].output[k]
                if trainer[i].output[k] == False:
                    all_stop += 1
            if all_stop == 4:
                trainer[i].stop_cou = stop_thres
            #print(trainer[i].output)
            player_flag[i] = True
            #for k,e in enumerate(player_info["sensor_offset"]):
            #    input[0][k] = input[0][k]/e[2]
            #    input[0][k+sensor_num] = input[0][k]/e[2]
            #print(trainer,i,input)
        if player_stop == player_num:
            iter += 1
            sort_list = []
            #score_list = []
            #max_score_idx = -1
            #max_score_tp = -1.0
            for i in range(0,player_num):
                sort_list.append([i,trainer[i].score])
                #score_list.append(trainer[i].score)
            sort_list = sorted(sort_list,key=lambda l:l[1], reverse=True)
            #for i,e in enumerate(score_list):
            #    if max_score_tp < e:
            #        max_score_tp = e
            #        max_score_idx = i
            #score_list = sorted(score_list,key=lambda l:l, reverse=True)
            
            print("iter:",iter,"current score",sort_list[0][1],"max score:",max_score)
            if max_score < sort_list[0][1]:
                max_score = sort_list[0][1]
                print("update max score:",max_score)
                #max_weights = copy.deepcopy(trainer[sort_list[0][0]].get_weights())
                #max_biases = copy.deepcopy(trainer[sort_list[0][0]].get_biases())
            print_hash(trainer,player_num)
            '''
            print("start")
            for k in range(0,player_num):
                tp_w=""
                tp_b=""
                _ = [i3 for i in trainer[k].get_weights() for i2 in i for i3 in i2]
                __ = [i2 for i in trainer[k].get_biases() for i2 in i]
                for i in _:
                    tp_w = tp_w + str(i)
                for i in __:
                    tp_b = tp_b + str(i)
                tp_w = hashlib.md5(tp_w.encode())
                tp_b = hashlib.md5(tp_b.encode())
                print(tp_w.hexdigest(),tp_b.hexdigest())
            '''
            #print(max_weights)
            new_weights_pool = mate_weights(trainer,sort_list,player_num,select_rate)
            #new_biases_pool = mate_biases(trainer,sort_list,player_num,select_rate)
            for i in range(0,player_num):
                
                if i == sort_list[0][0] or i == sort_list[1][0]:
                    trainer[i].score = 0
                    trainer[i].stop_cou = 0
                    trainer[i].high_score = 0
                    trainer[i].start = True
                    continue
                
                trainer[i].set_weights(new_weights_pool[i])
                #trainer[i].set_biases(new_biases_pool[i])
                trainer[i].score = 0
                trainer[i].stop_cou = 0
                trainer[i].high_score = 0
                trainer[i].start = True
                #print("set")
                

            
            #del new_biases_pool
            #del new_weights_pool
            player_stop = 0
            player_flag = [False for i in range(0,player_num)]
            for i in range(0,player_num):
                for k in range(0,4):
                    player_input_list[i][k] = False
            
            #for player in game.player:
            #    player.start = True
            #print("input")
            #print(iter,input_tp)
            #input_tp = [[] for k in range(0,player_num)]

def main():
    train()
            
main()