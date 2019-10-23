import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np
from player import Player
class TrackGameHelper():
    def __init__(self,map_file,obstacle_color,white_color,width,height,fps):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        obstacle,background, bin_list = self.get_map(map_file,obstacle_color)
        self.obstacle = obstacle
        self.background = background
        self.bin_list = bin_list
        self.obstacle_mask = pygame.mask.from_surface(obstacle)
        self.obstacle_rect = obstacle.get_rect()
        offset_x, offset_y = width/2-self.obstacle_rect.center[0], height/2-self.obstacle_rect.center[1]
        self.offset_x = offset_x
        self.offset_y =offset_y
        self.game_clock = pygame.time.Clock()
        self.player = []
        self.background_color = white_color
        self.fps = fps
        pygame.display.set_caption("track game")
    
    def load_sprite(file_path,scale=None):
        data = pygame.image.load(file_path)
        if scale != None:
            width, height = data.get_size()
            width, height = width*scale, height*scale
            data = pygame.transform.scale(data, (width, height))
        data = data.convert_alpha()
        return data    
    def set_player(self,car_file,car_file2,player_info,player_num):
        for i in range(0,player_num):
            print(car_file,car_file2)
            self.player.append(Player(car_file,car_file2,self.bin_list, self.obstacle_mask,self.offset_x, self.offset_y,player_info,self.fps,player_num))
    
    def get_map(self,map_file,obstacle_color,scale=None):
        bin_list_tp = []
        bin_list = []
        red, green, blue = obstacle_color
        img= Image.open(map_file).convert("RGBA")
        mode,size,data = img.mode, img.size, img.tobytes()
        background = pygame.image.fromstring(data, size, mode)
        data = img.getdata()
        new_data = []
        for item in data:
            if item[0] == red and item[1] == green and item[2] == blue:
                new_data.append((128, 128, 128, 255))
                bin_list_tp.append(1)
            else:
                new_data.append((0, 0, 0, 0))
                bin_list_tp.append(0)
        bin_list_tp = np.reshape(bin_list_tp, (size[1],size[0]))
        
        for i in range(0,len(bin_list_tp)):
            bin_list.append(list(bin_list_tp[i]))
        
        img.putdata(new_data)
        obstacle = pygame.image.fromstring(img.tobytes(), size, mode)
        if scale != None:
            width, height = obstacle.get_size()
            width, height = width*scale, height*scale
            obstacle = pygame.transform.scale(obstacle, (width, height))
            background = pygame.transform.scale(background, (width, height))
        obstacle, background = obstacle.convert_alpha(), background.convert_alpha()
        return obstacle, background, bin_list
        
    def run(self):
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
                    for player in self.player:
                        player.reset()
            for player in self.player:
                player.check_input()
            self.screen.blit(self.background, (self.offset_x, self.offset_y))
            for player in self.player:
                self.screen.blit(player.current_sprite, player.current_sprite_rect)
                for i in player.sensor_pos:
                    pygame.draw.line(self.screen,(128,128,128,0),(i[0],i[1]),(i[2],i[3]))
            #deg = (-1*(math.atan2(player.speed_y, player.speed_x) * (180 / math.pi)) + 360 ) %360
            #print(min(abs(abs(player.rotate - deg)),abs(player.rotate - deg+360)))
            pygame.display.update()
            self.game_clock.tick(self.fps)
            self.screen.fill(self.background_color)
            
    
    def run_step(self,player_input_list):
        for i,player in enumerate(self.player):
            if player_input_list == None:
                player.check_input()
            else:
                player.check_input(player_input_list[i])
        self.screen.blit(self.background, (self.offset_x, self.offset_y))
        for player in self.player:
            self.screen.blit(player.current_sprite, player.current_sprite_rect)
            for i in player.sensor_pos:
                pygame.draw.line(self.screen,(128,128,128,0),(i[0],i[1]),(i[2],i[3]))
        pygame.display.update()
        self.game_clock.tick(self.fps)
        self.screen.fill(self.background_color)   