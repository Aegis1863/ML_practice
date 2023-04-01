import pygame, sys, random
from pygame.locals import *

pygame.init()
win = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
