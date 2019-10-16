import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
import random
from track_game_helper import TrackGameHelper
from player import Player
from gene import Trainer

map_file = "data/track.png"
car_file = "data/car_blue.png"
car_file2 = "data/car_red.png"
obstacle_color = (128,128,128)
white_color = (255, 255, 255, 0)
width, height = 800, 600
fps = 30
user_input = True
player_num = 10
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
    while True:
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
            deg = (-1*(math.atan2(player.speed_y, player.speed_x) * (180 / math.pi)) + 360 ) %360
            forward = False
            if abs(player.rotate - deg) < 90:
                forward = True
            trainer[i].score += pow(player.speed_x,2)+pow(player.speed_y,2)
            #print(i,player.speed_x,player.speed_y,player.rotate,deg,abs(player.rotate - deg),forward)
            if player.collide == True or (forward == False and player_flag[i] == True) or (player.speed_x==0 and player.speed_y==0 and player_flag[i] == True):
                player.reset()
                trainer[i].start = False
                player_input_list[i] = [False,False,False,False]
                player_stop += 1
                continue
            input = [player.sensor_value_prev+player.sensor_value]
            sensor_num = len(player_info["sensor_offset"])
            #print(player.speed_x,player.speed_y)
            output = trainer[i].run(input)
            print(output)
            player_input_list[i]= output
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
                for i2 in range(0,math.ceil(node_count*select_rate)):
                    i3 = math.floor(random.random()*len(l)) # select layer
                    i4 = math.floor(random.random()*len(l[i3]))
                    i5 = math.floor(random.random()*len(l[i3][i4]))
                    l[i3][i4][i5] = random.random()*pow(10,36)
                new_pool.append(l)
            #for i in range(0,player_num):
                #trainer
                #print(a.shape,type(a))
            #for i in range(0,player_num):
            #    for k in range(0,4):
            #        player_input_list[i][k] = False
            #trainer = [Trainer(player_info,train_info) for i in range(0,player_num)]
            
            player_stop = 0
            player_flag = [False for i in range(0,player_num)]
            break

def main():
    train()
            
main()