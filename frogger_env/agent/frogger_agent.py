# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
# 
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

import numpy as np
from collections import deque

from frogger_env.agent.road import Vehicle

class Agent():

    """
    A simple agent

    """

    def __init__(self,
                 position,
                 goal,
                 action_map,
                 world_bounds,
                 heading: float = 0,
                 speed: float = 2,
                 radius: float = 0.5):
        self.spawn_position = np.array(position).astype('float')
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.speed = speed
        self.radius = radius
        self.crashed = False
        self.goal = goal
        self.action_map = action_map
        self.world_bounds = world_bounds
        self.act(0)
        self.nearest_to_goal = self.goal_distance
        self.instant_vel = np.array([0, 0])


    def act(self, action=0):
        """
        Store an action to be repeated.

        Params
        ======
        action (int): the input action
        """
        self.action = self.action_map[action]

    def step(self, dt):
        """
        Update the agent state given its actions.
        
        Parameters
        ----------
        dt (float): the integration time step 
        """
        # self.heading = 0
        # v = self.speed * np.array([np.cos(self.heading),
        #                            np.sin(self.heading)])
        temp = self.position + self.speed * dt * self.action
        if not (self.world_bounds[0] < temp[0] < self.world_bounds[2]) or \
                not (self.world_bounds[1] < temp[1] < self.world_bounds[3]):
            return

        self.instant_vel = self.speed * self.action
        self.heading = np.arctan2(self.instant_vel[1], self.instant_vel[0])
        self.nearest_to_goal = min(self.goal_distance, self.nearest_to_goal)
        self.position += self.speed * dt * self.action

    def check_collision(self, vehicles):
        """
        Check for collision with a list of vehicle.
        """
        if self.crashed:
            return

        for obs in vehicles:
            if np.linalg.norm(obs.position - self.position) > self.radius + Vehicle.DEFAULT_LENGTH*0.5:
                continue
            DeltaX = self.position[0] - max(obs.x_min, min(self.position[0], obs.x_max))
            DeltaY = self.position[1] - max(obs.y_min, min(self.position[1], obs.y_max))
            if (DeltaX * DeltaX + DeltaY * DeltaY) < self.radius*self.radius:
                self.crashed = True
                return

    @property
    def actions(self):
        return self.action_map

    @property
    def direction(self):
        return self.velocity_direction #np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self):
        return self.instant_vel

    @property
    def velocity_direction(self):
        current_speed = np.linalg.norm(self.instant_vel)
        if current_speed == 0:
            return np.zeros(2, )
        return self.instant_vel/current_speed

    @property
    def goal_direction(self):
        if (self.goal != self.position).any():
            return (self.goal - self.position) / np.linalg.norm(self.goal - self.position)
        else:
            return np.zeros((2,))

    @property
    def goal_distance(self):
        return np.linalg.norm(self.position - self.goal)

    @property
    def normalized_goal_distance(self):
        return max(self.goal_distance/np.linalg.norm(self.spawn_position - self.goal), 1)

    @property
    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
