# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
# 
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

from gym import spaces
import numpy as np
import math
import heapq
from frogger_env.envs.utils import *


class ObservationType(object):
    def __init__(self, env):
        self.env = env

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    def get_type(self):
        raise NotImplementedError()


class LidarScan(ObservationType):

    def __init__(self, env):
        super().__init__(env)
        self.fov = 360
        self.angle_resolution = self.env.observation_config["angle_resolution"]
        self.sensing_distance = self.env.observation_config["sensing_distance"]

    def observe(self):
        agent_pos = self.env.agent.position
        vehicles = self.env.road.vehicles
        nearest_vehicles_rect = []
        for v in vehicles:
            rect = Rectangle(v.x_max, v.x_min, v.y_max, v.y_min)
            _, dist = rect.nearest_to_point(agent_pos)
            if dist < self.sensing_distance:
                nearest_vehicles_rect.append(rect)

        lidar = []
        for angle in range(0, self.fov, self.angle_resolution):
            angle_radians = math.radians(angle)
            min_dist = self.sensing_distance
            for rects in nearest_vehicles_rect:
                _, dist = rects.ray_intersection(agent_pos, angle_radians)
                min_dist = min(dist, min_dist)
            lidar.append(min_dist)
        self.observations = lidar
        return lidar

    @property
    def type(self):
        return 'lidar'

    def space(self) -> spaces.Space:
        return spaces.Box(shape=[self.fov//self.angle_resolution, ],
                          low=0, high=self.sensing_distance, dtype=np.float32)


class OccupancyGrid(ObservationType):
    # GRID_SIZE = [[-3.5*3, 3.5*3], [-3.5*3, 3.5*3]]
    # GRID_STEP = [3, 3]
    GRID_SIZE = [[-3.5*3, 3.5*3], [-3.5*5, 3.5*5]]
    GRID_STEP = [3, 3]

    def __init__(self, env, grid_size=None, grid_step=None):
        super().__init__(env)
        if grid_size is not None:
            self.grid_size = np.array(grid_size)
        else:
            self.grid_size = np.array(self.GRID_SIZE)

        if grid_step is not None:
            self.grid_step = np.array(grid_step)
        else:
            self.grid_step = np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1]-self.grid_size[:, 0])
                                / self.grid_step), dtype=np.int)
        self.grid = np.zeros(grid_shape)
        self.grid_center = [len(self.grid)//2, len(self.grid[0])//2]
        self.rect_full_grid = Rectangle(self.grid_size[0][1], self.grid_size[0][0],
                                        self.grid_size[1][1], self.grid_size[1][0])
        # print(self.rect_full_grid.length, self.rect_full_grid.width)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add nearby traffic
        self.grid.fill(0)
        agent_pos = self.env.agent.position
        vehicles = self.env.road.vehicles
        for vehicle in vehicles:
            rect_car = Rectangle(vehicle.x_max - agent_pos[0],
                                 vehicle.x_min - agent_pos[0],
                                 vehicle.y_max - agent_pos[1],
                                 vehicle.y_min - agent_pos[1])
            if not self.rect_full_grid.rectangle_intersection(rect_car):
                continue
            # print("HERE")
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    cells_apartX = j - self.grid_center[1]
                    cells_apartY = i - self.grid_center[0]
                    rect_center = [cells_apartX * self.grid_step[1],
                                   cells_apartY * self.grid_step[0]]
                    rect_cell = Rectangle(rect_center[0] + self.grid_step[1]/2,
                                          rect_center[0] - self.grid_step[1]/2,
                                          rect_center[1] + self.grid_step[0]/2,
                                          rect_center[1] - self.grid_step[0]/2)
                    if rect_cell.rectangle_intersection(rect_car):
                        self.grid[i, j] = 1
        return self.grid

    @property
    def type(self):
        return 'occupancy_grid'


class Neighborhood(ObservationType):
    def __init__(self, env, normalize=True):
        super().__init__(env)
        self.nearest_neighbor = 5
        self.max_distance =  self.env.observation_config["sensing_distance"]
        self.max_distance_sqr = self.max_distance*self.max_distance
        self.normalize = normalize

    def normalize_obs(self, obs):
        # for the time being assume fixed range as below
        n_obs = []
        features_range = [
                [-self.max_distance, self.max_distance], # x
                [-self.max_distance, self.max_distance], # y
                [0, self.env.road.speed], # vx
                [-self.env.agent.speed, 0]] #vy
        for o in obs:
            for i, feature in enumerate(o):
                n_obs.append(-1 + (feature - features_range[i][0])*2./(features_range[i][1] - features_range[i][0]))
        return n_obs

    def space(self) -> spaces.Space:
        return spaces.Box(shape=[self.nearest_neighbor*4], low=-self.max_distance,
                          high=self.max_distance, dtype=np.float32)

    def observe(self) -> np.ndarray:
        agent_pos = self.env.agent.position
        vehicles = self.env.road.vehicles
        closest_vehicles = []
        for v in vehicles: # could use np.linalg.norm for faster comp
            dist = (agent_pos[0] - v.position[0])**2 + (agent_pos[1] - v.position[1])**2
            if dist < self.max_distance_sqr:
                 closest_vehicles.append([dist,v])
        observation = []
        if closest_vehicles:
            closest_vehicles.sort()
            for item in closest_vehicles:
                local_posX, local_posY = item[1].position - agent_pos
                local_velX, local_velY = item[1].velocity - self.env.agent.velocity
                observation.append([local_posX, local_posY, local_velX, local_velY])

        for _ in range(self.nearest_neighbor - len(closest_vehicles)):
            observation.append([self.max_distance,
                                self.max_distance, self.env.road.speed, -self.env.agent.velocity[1]])

        if self.normalize:
            observation = self.normalize_obs(observation)
        return observation

    @property
    def type(self):
        return 'neighborhood'
