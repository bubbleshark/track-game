import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
from track_game_helper import TrackGameHelper
from player import Player

map_file = "data/track2.png"
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
    "initial_x": 100,
    "initial_y": 248,
    "rotate": 270,
    "boost": 0.4,
    "slow_down": 0.1,
    "speed_max": 5,
    "sensor_offset": [[10,0,80,0],[12,340,60,45],[12,20,60,315],[9,316,40,90],[9,44,40,270]] # r,degree,r,degree
}
        
def main():
    game = TrackGameHelper(map_file,obstacle_color,white_color,width,height,fps)
    game.set_player(car_file,car_file2,player_info,player_num)
    game.run()
    '''
    player_input_list = [[False,False,False,False] for player in game.player]
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
                    player_input_list = [[False,False,False,False] for player in game.player]
                    for player in game.player:
                        player.reset()
        game.run_step(player_input_list)
        # for bot control
        
        for player in game.player:
            print(player.sensor_value)
        player_input_list= [None for player in game.player]
        for i,e in enumerate(player_input_list):
            player_input_list[i]= [True,False,False,False]
        '''
main()