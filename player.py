import math, random, sys
import pygame
from pygame.locals import *
from PIL import Image
import math
import numpy as np

class Player():
    #car_file,car_file2,self.bin_list, self.obstacle_mask,self.offset_x, self.offset_y,player_info,self.fps,player_num
    def __init__(self,status_file,status_file2,bin_list, obstacle_mask,offset_x,offset_y,player_info,fps,player_num):
        self.normal_sprite = self.load_sprite(status_file)
        self.collide_sprite = self.load_sprite(status_file2)
        self.current_sprite = None
        self.bin_list = bin_list
        self.rotate_step = player_info["rotate_step"]
        self.x = player_info["initial_x"]
        self.y = player_info["initial_y"]
        self.width = self.normal_sprite.get_width()
        self.height = self.normal_sprite.get_height()
        self.mid_x = self.x + self.width/2
        self.mid_y = self.y + self.height/2
        self.last_mid_x = self.x + self.width/2
        self.last_mid_y = self.y + self.height/2
        self.rotate = player_info["rotate"]
        self.boost_rotate = self.rotate
        self.rotate_sprite(normal=True)
        self.boost = player_info["boost"]
        self.slow_down = player_info["slow_down"]
        self.round_digits = 4
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.obstacle_mask = obstacle_mask
        self.player_mask = pygame.mask.from_surface(pygame.transform.rotate(self.normal_sprite,self.rotate))
        self.speed_x, self.speed_y = 0, 0
        self.speed_max = player_info["speed_max"]
        self.collide = False
        self.last_collide = self.collide
        self.sensor_offset = player_info["sensor_offset"]
        self.sensor_pos = [[] for i in range(0,len(self.sensor_offset))]
        self.sensor_value = [0 for i in range(0,len(self.sensor_offset))]
        self.sensor_value_prev = [0 for i in range(0,len(self.sensor_offset))]
        self.player_info = player_info
        self.start = False

    def reset(self):
        self.x =self.player_info["initial_x"]
        self.y = self.player_info["initial_y"]
        self.rotate = self.player_info["rotate"]
        self.boost_rotate = self.rotate
        self.mid_x = self.x + self.width/2
        self.mid_y = self.y + self.height/2
        self.speed_x, self.speed_y = 0, 0
        self.collide =False
        self.start = False
        for i in range(0,len(self.sensor_offset)):
            self.sensor_value_prev[i] = 0
        self.last_collide = self.collide
        self.width = self.normal_sprite.get_width()
        self.height = self.normal_sprite.get_height()
        self.mid_x = self.x + self.width/2
        self.mid_y = self.y + self.height/2
        self.last_mid_x = self.x + self.width/2
        self.last_mid_y = self.y + self.height/2
        self.rotate = self.player_info["rotate"]
        self.boost_rotate = self.rotate
        self.rotate_sprite(normal=True)
        self.boost = self.player_info["boost"]
        self.slow_down = self.player_info["slow_down"]
        self.round_digits = 4
        #self.offset_x = offset_x
        #self.offset_y = offset_y
        #self.obstacle_mask = obstacle_mask
        self.player_mask = pygame.mask.from_surface(pygame.transform.rotate(self.normal_sprite,self.rotate))
        self.speed_x, self.speed_y = 0, 0
        self.speed_max = self.player_info["speed_max"]
        self.collide = False
        self.last_collide = self.collide
        self.sensor_offset = self.player_info["sensor_offset"]
        self.sensor_pos = [[] for i in range(0,len(self.sensor_offset))]
        self.sensor_value = [0 for i in range(0,len(self.sensor_offset))]
        self.sensor_value_prev = [0 for i in range(0,len(self.sensor_offset))]
        #self.player_info = player_info
    
    def load_sprite(self,file_path,scale=None):
        data = pygame.image.load(file_path)
        if scale != None:
            width, height = data.get_size()
            width, height = width*scale, height*scale
            data = pygame.transform.scale(data, (width, height))
        data = data.convert_alpha()
        return data
    
    def check_collision(self):
        distance = (int(self.current_sprite_rect.x - self.offset_x), int(self.current_sprite_rect.y - self.offset_y))
        self.player_mask = pygame.mask.from_surface(pygame.transform.rotate(self.normal_sprite,self.rotate))
        is_normal = self.obstacle_mask.overlap(self.player_mask, distance)
        if is_normal is None: #no collide
            self.collide = False
            is_normal = True
        else: # collide
            self.speed_x, self.speed_y = 0, 0
            self.collide = True
            is_normal = False
        self.rotate_sprite(is_normal)
    
    def strict_collide(self):
        if self.last_collide == True and self.collide == True:
            self.speed_x=0
            self.speed_y=0
            self.x = self.last_x
            self.y = self.last_y

    def rotate_sprite(self,normal):
        if normal == True:
            self.current_sprite = pygame.transform.rotate(self.normal_sprite,self.rotate)
        else:
            self.current_sprite = pygame.transform.rotate(self.collide_sprite,self.rotate)
        self.current_sprite_rect = self.current_sprite.get_rect()
        self.current_sprite_rect.center = (self.mid_x,self.mid_y)      
   
    def get_rotate_offset(self,x,y,r,degree):
        rad = (-1*degree) * math.pi / 180
        nx = r*math.cos(rad)
        ny = r*math.sin(rad)
        nr = math.sqrt(nx*nx+ny*ny)
        scaler = float(nr)/r
        nx*=scaler
        ny*=scaler
        return x+nx,y+ny
        
    def get_rotate(self,cx, cy, r,sensor_degree, degree):
        rad = (-1*degree+sensor_degree) * math.pi / 180
        ox = r*math.cos(rad)
        oy = r*math.sin(rad)
        x = cx + ox
        y = cy + oy
        return round(x),round(y)
    
    def update_snesor(self):
        for i,e in enumerate(self.sensor_offset):
            pos = self.get_rotate(self.mid_x,self.mid_y,e[0],e[1],self.rotate)
            x, y = pos[0], pos[1]
            x2,y2 = self.get_rotate_offset(x,y,e[2],self.rotate+e[3])
            self.sensor_pos[i] = [x,y,x2,y2]
    
    def check_collide_map(self,x,y):
        if self.bin_list[y][x]==1:
            return True
        return False

    def get_sensor_value(self):
        for i,e in enumerate(self.sensor_pos):
            x1,y1,x2,y2=e[0],e[1],e[2],e[3]
            eps = 0.000001
            a=(y1-y2)/(x1-x2-eps)
            b=y1-x1*a
            a2=(x1-x2)/(y1-y2-eps)
            b2=x1-y1*a2
            cou_a, cou_b = None, None
            if abs(a) > abs(a2):
                step = 1
                if y1 > y2:
                    step*=-1
                tp_x = x1
                tp_y = y1
                collid_flag = False
                
                while math.sqrt(pow((tp_x-x2),2)+pow((tp_y-y2),2)) >= 1.0:
                    if self.check_collide_map(round(tp_x),round(tp_y)) is True:
                        collid_flag = True
                        break
                    tp_y += step
                    tp_x = a2*tp_y + b2
                if collid_flag is False:
                    if self.check_collide_map(round(x2),round(y2)):
                        collid_flag = True
                        tp_x, tp_y = x2, y2
                if self.start == False:
                    self.sensor_value_prev[i] = 0
                else:
                    self.sensor_value_prev[i] = self.sensor_value[i]
                if collid_flag is True:
                    self.sensor_value[i] = math.sqrt(pow((tp_x-x1),2)+pow((tp_y-y1),2))
                else:
                    self.sensor_value[i] = self.sensor_offset[i][2]
            else:
                step = 1
                if x1 > x2:
                    step*=-1
                tp_x = x1
                tp_y = y1
                collid_flag = False
                while math.sqrt(pow((tp_x-x2),2)+pow((tp_y-y2),2)) >= 1.0:
                    if self.check_collide_map(round(tp_x),round(tp_y)) is True:
                        collid_flag = True
                        break
                    tp_x += step
                    tp_y = a*tp_x + b
                if collid_flag is False:
                    if self.check_collide_map(round(x2),round(y2)):
                        collid_flag = True
                        tp_x, tp_y = x2, y2
                if self.start == False:
                    self.sensor_value_prev[i] = 0
                else:
                    self.sensor_value_prev[i] = self.sensor_value[i]
                if collid_flag is True:
                    self.sensor_value[i] = math.sqrt(pow((tp_x-x1),2)+pow((tp_y-y1),2))
                else:
                    self.sensor_value[i] = self.sensor_offset[i][2]
    
    def check_user_input(self):
        input_list = [False, False, False, False] # left, right, up, down
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            input_list[0] = True
        if key[pygame.K_RIGHT]:
            input_list[1] = True
        if key[pygame.K_UP]:
            input_list[2] = True
        if key[pygame.K_DOWN]:
            input_list[3] = True
        return input_list
        
    def check_input(self,input_list=None):
        boost_flag =False
        if input_list is None:
            input_list = self.check_user_input()
        if input_list[0] is True:
            self.rotate += self.rotate_step
            self.rotate %= 360
        if input_list[1] is True:
            self.rotate += -1*self.rotate_step
            self.rotate %= 360
        if input_list[2] is True:
            self.boost_rotate = self.rotate
            self.speed_x += self.boost*math.cos(self.rotate*math.pi/180)
            self.speed_y -= self.boost*math.sin(self.rotate*math.pi/180)
            boost_flag = True
        if input_list[3] is True:
            self.boost_rotate = self.rotate
            self.speed_x -= self.boost*math.cos(self.rotate*math.pi/180)
            self.speed_y += self.boost*math.sin(self.rotate*math.pi/180)
            boost_flag = True    
        # resistance
        z = None
        if abs(self.speed_x) or abs(self.speed_y):
            z = math.sqrt(math.pow(self.speed_x,2) + math.pow(self.speed_y,2))
        if self.speed_x > 0:
            self.speed_x -= self.slow_down*abs(self.speed_x)/z
            if self.speed_x < 0:
                self.speed_x = 0
        elif self.speed_x < 0:
            self.speed_x += self.slow_down*abs(self.speed_x)/z
            if self.speed_x > 0:
                self.speed_x = 0
        if self.speed_y > 0:
            self.speed_y -= self.slow_down*abs(self.speed_y)/z
            if self.speed_y < 0:
                self.speed_y = 0
        elif self.speed_y < 0:
            self.speed_y += self.slow_down*abs(self.speed_y)/z
            if self.speed_y > 0:
                self.speed_y = 0

        last_collide = self.collide
        self.last_mid_x, self.last_mid_y = self.mid_x,self.mid_y
        tp_speed = math.sqrt(pow(self.speed_x,2)+pow(self.speed_y,2))
        if tp_speed > self.speed_max:
            scale = float(self.speed_max)/tp_speed
            self.speed_x *= scale
            self.speed_y *= scale
        self.speed_x = round(self.speed_x,self.round_digits)
        self.speed_y = round(self.speed_y,self.round_digits)
        self.mid_x += self.speed_x
        self.mid_y += self.speed_y
        self.x += self.speed_x
        self.y += self.speed_y
        self.last_collide = self.collide
        self.check_collision()
        self.update_snesor()
        self.get_sensor_value()
        value_list = []
        for i in self.sensor_value:
            value_list.append(i)