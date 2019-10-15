import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
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
player_num = 1
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
        
def train():
    game = TrackGameHelper(map_file,obstacle_color,white_color,width,height,fps)
    game.set_player(car_file,car_file2,player_info,player_num)
    player_input_list= []
    for i in range(0,player_num):
        player_input_list.append([False,False,False,False])
    print(player_input_list)
    trainer = [Trainer(player_info,train_info) for i in range(0,player_num)]
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
            input = [player.sensor_value_prev+player.sensor_value]
            sensor_num = len(player_info["sensor_offset"])
            #for k,e in enumerate(player_info["sensor_offset"]):
            #    input[0][k] = input[0][k]/e[2]
            #    input[0][k+sensor_num] = input[0][k]/e[2]
            print(trainer,i,input)
            output = trainer[i].run(input)

            if player.collide == True:
                player.reset()
            player_input_list[i]= output

def main():
    train()
            
main()