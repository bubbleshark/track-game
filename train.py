import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
import random
from track_game_helper import TrackGameHelper
from player import Player
from gene import Trainer, mate_weights, mate_biases
from mem_top import mem_top

map_file = "data/track.png"
car_file = "data/car_blue.png"
car_file2 = "data/car_red.png"
obstacle_color = (128,128,128)
white_color = (255, 255, 255, 0)
width, height = 800, 600
fps = 30
user_input = True
player_num = 5
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
select_rate = 0.2
        
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
    while True:
        print(cou,mem_top())
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
            #print(i)
            deg = (-1*(math.atan2(player.speed_y, player.speed_x) * (180 / math.pi)) + 360 ) %360
            forward = False
            if abs(player.rotate - deg) < 90:
                forward = True
            trainer[i].score += pow(player.speed_x,2)+pow(player.speed_y,2)
            #print(i,player.speed_x,player.speed_y,player.rotate,deg,abs(player.rotate - deg),forward)
            if player.collide == True or (forward == False and player_flag[i] == True) or (player.speed_x==0 and player.speed_y==0 and player_flag[i] == True):
                player.reset()
                #rint("stop",i)
                trainer[i].start = False
                player_input_list[i] = [False,False,False,False]
                player_stop += 1
                continue
            input = [player.sensor_value_prev+player.sensor_value]
            sensor_num = len(player_info["sensor_offset"])
            #print(player.speed_x,player.speed_y)
            trainer[i].run(input)
            for k in range(0,4):
                player_input_list[i][k] = trainer[i].output[k]
            player_flag[i] = True
            #for k,e in enumerate(player_info["sensor_offset"]):
            #    input[0][k] = input[0][k]/e[2]
            #    input[0][k+sensor_num] = input[0][k]/e[2]
            #print(trainer,i,input)
        if player_stop == player_num:
            sort_list = []
            for i in range(0,player_num):
                sort_list.append([i,trainer[i].score])
            sort_list = sorted(sort_list,key=lambda l:l[1], reverse=True)
            #print("score",sort_list[0][1],max_score)
            if max_score < sort_list[0][1]:
                max_score = sort_list[0][1]
                max_weights = trainer[sort_list[0][0]].get_weights()
                max_biases = trainer[sort_list[0][0]].get_biases()
                
            new_weights_pool = mate_weights(trainer,sort_list,player_num,select_rate,max_weights)
            new_biases_pool = mate_biases(trainer,sort_list,player_num,select_rate,max_biases)
            for i in range(0,player_num):
                trainer[i].set_weights(new_weights_pool[i])
                trainer[i].set_biases(new_biases_pool[i])
                trainer[i].score = 0
                trainer[i].start = True
                #print("set")
            #del new_biases_pool
            #del new_weights_pool
            player_stop = 0
            player_flag = [False for i in range(0,player_num)]
            for i in range(0,player_num):
                for k in range(0,4):
                    player_input_list[i][k] = False

def main():
    train()
            
main()