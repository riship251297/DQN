# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
#
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

import numpy as np
from typing import Tuple
from gym.envs.registration import register
from frogger_env.envs.abstract import AbstractEnv
from frogger_env.agent.road import Road, Vehicle
from frogger_env.agent.frogger_agent import Agent


class CrossingEnv_V1(AbstractEnv):
    """
    A modified frogger environment.

    The agent has to cross the highway and reach the other side.
    """

    """The reward received when reaching the other side of the highway."""
    GOAL_REWARD = 1.0

    """ The reward received when colliding with a vehicle. """
    COLLISION_REWARD = -0.3

    """ The reward associated with how much aligned the agent is to its goal. """
    GOAL_ALIGNMENT_REWARD = 0.01

    ACTIONS_MAP = np.array([[0, 0],
                           [0, 1],
                           [0, -1],
                           [-1, 0],
                           [1, 0]])

    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "lidar":
                    {
                        "sensing_distance": 10,
                        "angle_resolution": 12,
                        "frame_history": 4,
                        "flatten_observation": True,
                        "include_goal_distance": True,
                        "include_goal_local_coodinates": True
                    },
                "occupancy_grid":
                    {
                        "frame_history": 1,
                        "flatten_observation": True,
                        "include_goal_distance": True,
                        "include_goal_local_coodinates": True
                    },

            },
            "observation_type": "lidar",
            "world_bounds": [0., 0., 50., 50],
            "lanes_count": 4,
            "vehicles_count": 20,  # upper bound
            "duration": 60,
            "vehicle_spacing": 2.5,
            "vehicle_speed": 3.5,
            "vehicle_width": 2.,
            "random_init": 1,
            "bidirectional": 1
        })
        return config

    def _reset(self):
        self._create_road()
        self._create_agent()

    def _create_road(self):
        """
        Create a road composed of straight adjacent lanes and populate it with vehicles.
        """
        self.road = Road(vehicles=[], lanes=[], np_random=self.np_random,
                         bidirectional=self.config["bidirectional"])
        self.road.generate_lanes(self.config["lanes_count"], length=50.)
        for _ in range(self.config["vehicles_count"]):
            self.road.generate_random_vehicle(speed=self.config["vehicle_speed"],
                                              lane_id=None,
                                              spacing=self.config["vehicle_spacing"], width=self.config["vehicle_width"])

    def _create_agent(self):
        """
        Create the agent. Could be spawn randomly.
        """
        self.agent_spawn = [10 + np.random.rand()*30.,  self.road.get_first_lane_Y() -4] if self.config["random_init"]\
        else [25,  self.road.get_first_lane_Y() - 4]
        self.goal = [25, self.road.get_last_lane_Y() + 8]
        self.agent = Agent(np.array(self.agent_spawn), radius=0.75, goal=self.goal,
                           action_map=self.ACTIONS_MAP, speed=2,
                           world_bounds=self.config["world_bounds"])
        self.lower_boundary = [self.agent_spawn[0], self.agent_spawn[1] - 2]

    def _reward(self, action):
        """
        The reward is defined to encourage the agent move towards the goal and cross the highway,
         while avoiding collisions.
        """
        reward = self.COLLISION_REWARD * int(self.agent.crashed) \
            + self.GOAL_REWARD * int(self.agent.goal_distance < 5) \
            + self.GOAL_ALIGNMENT_REWARD * np.dot(self.agent.velocity_direction,
                                                  self.agent.goal_direction)
        return reward

    def _is_terminal(self):
        """
        The episode is over if the agent collides, or the episode duration is met, 
        or the agent is close to the goal.
        """
        return self.agent.crashed or \
            (self.time >= self.config["duration"]
             and not self.config["manual_control"]) or \
            self.agent.goal_distance < 5 or \
            self.agent.position[1] < self.lower_boundary[1]


register(
    id='frogger-v1',
    entry_point='frogger_env.envs:CrossingEnv_V1',
)
