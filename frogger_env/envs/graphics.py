# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
#
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

from typing import TYPE_CHECKING
import time
import os
import math
import numpy as np
import pygame
if TYPE_CHECKING:
    from frogger_env.envs import AbstractEnv


class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    def __init__(self, env):
        self.env = env
        pygame.init()
        self.image_mode = self.env.config["offscreen_rendering"]
        if self.image_mode:
            self.screen = pygame.Surface([self.env.config["screen_width"],
                                          self.env.config["screen_width"]])
        else:
            self.screen = pygame.display.set_mode([self.env.config["screen_width"],
                                                   self.env.config["screen_width"]])

        self.screen.fill((255, 255, 255))
        self.origin = [self.env.config["world_bounds"][0], self.env.config["world_bounds"][1]]
        self.world_width = self.env.config["world_bounds"][2] - self.env.config["world_bounds"][0]
        self.world_height = self.env.config["world_bounds"][3] - self.env.config["world_bounds"][1]
        self.scaling = self.env.config["screen_width"]/self.world_width
        self.screen_height = self.env.config["screen_width"]
        print("Origin: ", self.origin)
        print("world_width: ", self.world_width)
        print("world_height: ", self.world_height)
        print("Screen: ", self.screen)
        # exit()
        # self.env.config.get("scaling", self.env.config["screen_width"]/self.world_width)
        self.manual = self.env.config["manual_control"]
        self.enabled = True
        self.unpaused = True
        self.toggleVisualization = True
        self.frame = 0
        self.directory = None
        self.agent_radius = self.convert_length_to_screen(self.env.agent.radius)
        self.clock = pygame.time.Clock()

    def get_events(self):
        for event in pygame.event.get():
            # issues with quit?
            '''if event.type == pygame.QUIT:
                self.env.close()
                break'''
            if event.type == pygame.KEYDOWN:
                '''if event.key == pygame.K_ESCAPE:
                    self.env.close()
                    break'''
                if event.key == pygame.K_SPACE:
                    self.unpaused = not self.unpaused
                if event.key == pygame.K_v:
                    self.toggleVisualization = not self.toggleVisualization
                '''if self.manual:
                    if event.key == pygame.K_w:
                        self.env.agent.act(1)
                    elif event.key == pygame.K_s:
                        self.env.agent.act(2)
                    elif event.key != pygame.K_SPACE and event.key != pygame.K_v:
                        self.env.agent.act(0)'''

    def convert_coordinates_to_screen(self, xy):
        pX = int(xy[0] * self.scaling)
        pY = int(self.screen_height - xy[1] * self.scaling)
        return pX, pY

    def convert_length_to_screen(self, length):
        return int(self.scaling * length)

    def draw_lidar_observation(self):
        agent_screen_pos = self.convert_coordinates_to_screen(self.env.agent.position)
        lidar_observation = self.env.observation_type.observations
        for lidar, angle in zip(lidar_observation, range(0,
                                self.env.observation_type.fov,
                                self.env.observation_type.angle_resolution)):
            angle_radians = math.radians(angle)
            line_origin = np.array(self.env.agent.position)
            ray_dir = np.array([np.cos(angle_radians), np.sin(angle_radians)])
            line_end = line_origin + lidar * ray_dir
            line_end_pos = self.convert_coordinates_to_screen(line_end)
            line_color = (255, 0, 0) if lidar < self.env.observation_type.sensing_distance \
                else (0, 255, 0)
            pygame.draw.line(self.screen,
                             line_color,
                             agent_screen_pos, line_end_pos,
                             2)

    def draw_grid_observation(self):
        # agent_screen_pos = self.convert_coordinates_to_screen(self.env.agent.position)
        grid_observation = self.env.observation_type
        grid = grid_observation.grid
        grid_step = grid_observation.grid_step
        grid_center = grid_observation.grid_center
        surface = pygame.Surface([self.env.config["screen_width"], self.env.config["screen_width"]],
                                 pygame.SRCALPHA)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                cells_apartX = j - grid_center[1]
                cells_apartY = i - grid_center[0]
                draw_location_local = [cells_apartX * grid_step[1],
                                       cells_apartY * grid_step[0]]
                draw_location = [draw_location_local[0] + self.env.agent.position[0],
                                 draw_location_local[1] + self.env.agent.position[1]]

                draw_location_screen = self.convert_coordinates_to_screen(draw_location)
                width = self.convert_length_to_screen(grid_step[1]-0.5)
                length = self.convert_length_to_screen(grid_step[0]-0.5)
                if grid[i][j] == 1:
                    color = [255, 0, 0, 128]
                else:
                    color = [100, 100, 100, 128]
                pygame.draw.rect(surface,
                                 color,
                                 [draw_location_screen[0] - length/2,
                                  draw_location_screen[1] - width/2,
                                  width,
                                  length])
        self.screen.blit(surface, [0, 0])

    def draw_lanes(self):
        for i, lane in enumerate(self.env.road.lanes):
            lX, lY = self.convert_coordinates_to_screen(lane.start + [0, lane.width/2])
            # print(lX, lY)
            width = self.convert_length_to_screen(lane.width)
            length = self.convert_length_to_screen(lane.end[0] - lane.start[0])
            if i % 2 == 0:
                color = [127, 127, 127]  #color = [255, 253, 208]
            else:
                color = [195, 195, 195]  #color = [155, 153, 108]
            pygame.draw.rect(self.screen,
                             color,
                             [lX, lY, length, width])

        for i, lane in enumerate(self.env.road.lanes):
            pygame.draw.line(self.screen,
                             (50, 0, 0),
                             self.convert_coordinates_to_screen(lane.start + [0, lane.width/2]),
                             self.convert_coordinates_to_screen(lane.end + [0, lane.width/2]),
                             2)
            pygame.draw.line(self.screen,
                             (50, 0, 0),
                             self.convert_coordinates_to_screen(lane.start - [0, lane.width/2]),
                             self.convert_coordinates_to_screen(lane.end - [0, lane.width/2]),
                             2)

    def draw_goal(self):
        line1_p1 = self.convert_coordinates_to_screen(self.env.agent.goal + np.array([1, 1]))
        line1_p2 = self.convert_coordinates_to_screen(self.env.agent.goal + np.array([-1, -1]))
        line2_p1 = self.convert_coordinates_to_screen(self.env.agent.goal + np.array([-1, 1]))
        line2_p2 = self.convert_coordinates_to_screen(self.env.agent.goal + np.array([1, -1]))
        # goalX, goalY = self.convert_coordinates_to_screen(self.env.agent.goal)
        pygame.draw.line(self.screen,
                         (136, 0, 20), #(0, 120, 0),
                         line1_p1, line1_p2,
                         10)
        pygame.draw.line(self.screen,
                         (136, 0, 20), #(0, 120, 0),
                         line2_p1, line2_p2,
                         10)

    def draw_spawn(self):
        line1_p1 = self.convert_coordinates_to_screen(self.env.agent_spawn + np.array([0.5, 0.5]))
        line1_p2 = self.convert_coordinates_to_screen(self.env.agent_spawn + np.array([-0.5, -0.5]))
        line2_p1 = self.convert_coordinates_to_screen(self.env.agent_spawn + np.array([-0.5, 0.5]))
        line2_p2 = self.convert_coordinates_to_screen(self.env.agent_spawn + np.array([0.5, -0.5]))
        pygame.draw.line(self.screen,
                         (120, 120, 0),
                         line1_p1, line1_p2,
                         5)
        pygame.draw.line(self.screen,
                         (120, 120, 0),
                         line2_p1, line2_p2,
                         5)

    def draw_vehicles(self):
        for v in self.env.road.vehicles:
            left = v.position[0] - v.length/2
            top = v.position[1] + v.width/2
            vX, vY = self.convert_coordinates_to_screen((left, top))
            carwidth = self.convert_length_to_screen(v.width)
            carlength = self.convert_length_to_screen(v.length)
            pygame.draw.rect(self.screen, v.color, [vX, vY, carlength, carwidth])

    def draw_agent(self):

        agent_screen_pos = self.convert_coordinates_to_screen(self.env.agent.position)
        pygame.draw.circle(self.screen,
                           (0, 0, 255),
                           agent_screen_pos,
                           self.agent_radius)

        '''
        agent_point1 = self.convert_coordinates_to_screen(self.env.agent.position + np.array([0, 1]) * self.agent_radius)
        agent_point2 = self.convert_coordinates_to_screen(self.env.agent.position + np.array([-0.65, -0.65]) * self.agent_radius)
        agent_point3 = self.convert_coordinates_to_screen(self.env.agent.position + np.array([0.65, -0.65]) * self.agent_radius)
        pygame.draw.polygon(self.screen,
                            (255, 255, 255),
                            [agent_point1, agent_point2, agent_point3])'''

    def display(self, mode):
        """Display the agent, road, and vehicles."""
        if not self.enabled:
            return
        self.get_events()
        while not self.unpaused:
            self.get_events()

        self.screen.fill((176, 193, 168)) #((205, 225, 205))
        self.draw_goal()
        self.draw_lanes()
        self.draw_vehicles()
        self.draw_spawn()
        if self.toggleVisualization:
            if self.env.observation_type.type == "lidar":
                self.draw_lidar_observation()
            elif self.env.observation_type.type == "occupancy_grid":
                self.draw_grid_observation()
        self.draw_agent()

        if not self.image_mode:
            pygame.display.flip()
            self.clock.tick(self.env.config["simulation_frequency"])

    def close(self):
        pygame.quit()

    def get_image(self):
        imgdata = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
        return imgdata
