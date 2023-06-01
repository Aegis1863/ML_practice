import pygame, sys, random
import box_control
from pygame.locals import *

win_width = 800
win_height = 800
box_num = 100
# 每行每列被划分出多少格，一共box_num*box_num格，一个元胞长宽win_height/box_num个像素
life_num = 500
change_start = True

pygame.init()
boxs = box_control.init_boxs(box_num, win_width, win_height, life_num)
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('zhan')
clock = pygame.time.Clock()


def flush_box(box_color, rect):
    pygame.draw.rect(win, box_color, rect) # 在什么上面填充，填充颜色，填充位置


def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


while True:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and event.key == K_SPACE:
            change_start = not change_start

    if change_start:
        for x in range(0, box_num):
            for y in range(0, box_num):
                if boxs[x][y]['value'] == 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)
                    # color = get_random_color()
                flush_box(color, # 生是黑色，死是白色
                          pygame.Rect(
                              boxs[x][y]['x'], # 横坐标
                              boxs[x][y]['y'], # 纵坐标
                              boxs[x][y]['width'], # 横宽
                              boxs[x][y]['height'], # 纵宽
                          ))
        box_control.change_boxs(boxs) # boxs包含全部网格状态

    pygame.display.update()