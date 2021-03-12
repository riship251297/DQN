# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
# 
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

import copy
import os

import gym
from gym import spaces
from gym.utils import seeding
from collections import deque

import numpy as np

from frogger_env.agent.frogger_agent import Agent
from frogger_env.agent.road import Vehicle, Road
from frogger_env.envs.graphics import EnvViewer
from gym import spaces
from frogger_env.envs.observation import *


class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks.

    The action space is fixed, but the observation space and reward function must be implemented.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, config=None):
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)
        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.agent = None
        self.agent_spawn = [25, 16]

        # Spaces
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.reset()

    def agent(self):
        return self.agent

    @classmethod
    def default_config(cls):
        """
        Default environment configuration.
        Can be overloaded in environment implementations.

        Returns
        ----------
        a configuration dict
        """
        return {
            "simulation_frequency": 20,  # [Hz]
            "policy_frequency": 2,  # [Hz]
            "screen_width": 600,  # [px]
            "screen_height": 400,  # [px]
            "scaling": 1.,  # this may need to be determined manually based on world width/height
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
            "bidirectional": 0
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config: dict):
        if config:
            self.config.update(config)

    def define_spaces(self):
        """
        Set the types and spaces of observation and action from config.
        """
        if self.config["observation_type"] == 'lidar':
            self.observation_config = self.config["observation"]["lidar"]
            self.observation_type = LidarScan(self)
            box = self.observation_type.space()
            if not self.observation_config["flatten_observation"]:
                raise NotImplementedError()
            else:
                goal_offset = int(self.observation_config["include_goal_distance"]) \
                    + int(self.observation_config["include_goal_local_coodinates"]) * 2
                self.observation_space = spaces.Box(shape=[self.observation_config["frame_history"]*box.shape[0] + goal_offset],
                                                    low=0, high=1,
                                                    dtype=box.dtype)
        elif self.config["observation_type"] == 'occupancy_grid':
            self.observation_config = self.config["observation"]["occupancy_grid"]
            self.observation_type = OccupancyGrid(self)
            box = self.observation_type.space()
            if not self.observation_config["flatten_observation"]:
                raise NotImplementedError()
            else:
                goal_offset = int(self.observation_config["include_goal_distance"]) \
                    + int(self.observation_config["include_goal_local_coodinates"]) * 2
                self.observation_space = spaces.Box(shape=[self.observation_config["frame_history"]*box.shape[0]*box.shape[1]+ goal_offset],
                                                    low=0, high=1,
                                                    dtype=box.dtype)
        elif self.config["observation_type"] == 'neighborhood':
            self.observation_config = self.config["observation"]["neighborhood"]
            self.observation_type = Neighborhood(self)
            box = self.observation_type.space()
            if not self.observation_config["flatten_observation"]:
                raise NotImplementedError()
            else:
                goal_offset = int(self.observation_config["include_goal_distance"]) \
                    + int(self.observation_config["include_goal_local_coodinates"]) * 2
                self.observation_space = spaces.Box(shape=[self.observation_config["frame_history"]*box.shape[0]+goal_offset],
                                                    low=-1, high=1,
                                                    dtype=box.dtype)
        if self.agent:
            self.action_set = self.agent.actions #(self.ACTIONS_ALL if self.config["full_actions"]
                            #else self.ACTIONS_MINIMAL)
            self.action_space = spaces.Discrete(len(self.action_set))

    def _reward(self, action):
        """
        Return the reward associated with performing a given action and
        ending up in the current state.

        Parameters
        ----------
        action: the last action performed
        
        Returns
        ----------
        the reward
        """
        raise NotImplementedError

    def _is_terminal(self):
        """
        Check whether the current state is a terminal state

        Returns
        ----------
        is the state terminal
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.define_spaces()  # First, to set the agent class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the agent once the scene is created
        self.observation_history = deque(maxlen=self.observation_config["frame_history"])
        observation = self.observation_type.observe()
        for _ in range(self.observation_config["frame_history"]):
            self.observation_history.append(observation)
        obs = np.array(self.observation_history)
        if self.observation_config["flatten_observation"]:
            obs = obs.flatten()
            if self.config["observation_type"] == "lidar":
                obs = obs/self.observation_config["sensing_distance"]
            if self.observation_config["include_goal_distance"]:
                obs = np.concatenate([obs, [self.agent.normalized_goal_distance]])
            if self.observation_config["include_goal_local_coodinates"]:
                obs = np.concatenate([obs, self.agent.goal_direction])
        else:
            raise NotImplementedError()
        return obs

    def _reset(self) -> None:
        """
        Reset the scene: road and agent.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the agent; all vehicles on the road perform their default behavior
        for several simulation timesteps until the next decision making step.

        Parameters
        ----------
        action: the action performed by the agent
        
        Returns
        ----------
        a tuple (observation, reward, terminal, info)
        """
        if self.agent is None:
            raise NotImplementedError("The agent and road must be initialized in the environment implementation")

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.steps += 1
        self._simulate(action)

        self.observation_history.append(self.observation_type.observe())
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "crashed": self.agent.crashed,
            "action": action,
            "TimeLimit.truncated": self.time >= self.config["duration"]
        }
        obs = np.array(self.observation_history)
        if self.observation_config["flatten_observation"]:
            obs = obs.flatten()
            if self.config["observation_type"] == "lidar":
                obs = obs/self.observation_config["sensing_distance"]
            if self.observation_config["include_goal_distance"]:
                obs = np.concatenate([obs, [self.agent.normalized_goal_distance]])
            if self.observation_config["include_goal_local_coodinates"]:
                obs = np.concatenate([obs, self.agent.goal_direction])
        else:
            raise NotImplementedError()
        # self.render()
        # print(obs)
        return obs, reward, terminal, info

    def _simulate(self, action):
        """
        Perform several steps of simulation with constant action.
        """
        if self.config["manual_control"]:
            steps = 1
        else:
            steps = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        self.agent.act(action)
        for _ in range(steps):
            # Forward action to the agent
            self.agent.step(1 / self.config["simulation_frequency"])
            # update the vehicles
            self.road.step(1 / self.config["simulation_frequency"])
            self.agent.check_collision(self.road.vehicles)

            self.time += 1 / self.config["simulation_frequency"]

            # Automatically render intermediate simulation steps if online rendering 
            self._automatic_rendering()
            # self.observation_history.append(self.observation_type.observe())
            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def render(self, mode='human'):
        """
        Render the environment.
        Create a viewer if none exists, and use it to render an image.

        Parameters
        ----------
        mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        if self.should_update_rendering:
            self.viewer.display(mode)

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

        self.should_update_rendering = False

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self):
        """
        Return the list of available actions
        """
        return self.action_set

    def _automatic_rendering(self):
        """
        Automatically render the intermediate frames while an action is still ongoing.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True
            self.render(self.rendering_mode)

    def __deepcopy__(self, memo):
        """
        Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
