# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
# 
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

from enum import Enum
import numpy as np


class LineType(Enum):
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2

# LineType = Enum(NONE, STRIPED, CONTINUOUS)


class Lane():

    """A simple class for road lanes."""

    DEFAULT_WIDTH = 5
    DEFAULT_LANE_SPACE = 20

    def __init__(self,
                 start,
                 end,
                 type,
                 direction=1,
                 width=DEFAULT_WIDTH):

        self.start = start
        self.end = end
        self.direction = direction
        self.type = type
        self.width = width


class Vehicle():
    """
    A simple vehicle on a road
    """

    DEFAULT_SPEEDS = [3.5, 5.5]
    DEFAULT_LENGTH = 5.
    DEFAULT_WIDTH = 2.

    ALLOWED_COLOR = [[255, 62, 40], [108, 143, 255]]  # [[50, 0, 100], [100, 50, 50]]

    def __init__(self,
                 position,
                 lane,
                 speed=0,
                 length=DEFAULT_LENGTH,
                 width=DEFAULT_WIDTH):
        self.position = np.array(position).astype('float')
        self.lane = lane
        self.speed = speed
        self.velocity = [self.speed, 0]
        self.width = width
        self.length = length
        self.color = Vehicle.ALLOWED_COLOR[np.random.choice(len(Vehicle.ALLOWED_COLOR))]
        # AABB representation
        self.x_min = self.position[0] - self.length/2.
        self.x_max = self.position[0] + self.length/2.
        self.y_min = self.position[1] - self.width/2.
        self.y_max = self.position[1] + self.width/2.

 
    def step(self, dt):
        """
        Update the vehicle state given its speed.
        """
        self.position += np.array([self.speed, 0]) * dt * self.lane.direction
        if self.lane.direction == 1:
            if self.position[0] - self.length > self.lane.end[0]:  # reset the vehicle
                self.position[0] = self.lane.start[0] - self.length
        else:
            if self.position[0] - self.length < self.lane.start[0]:  # reset the vehicle
                self.position[0] = self.lane.end[0] + self.length
        self.x_min = self.position[0] - self.length/2.
        self.x_max = self.position[0] + self.length/2.
        self.y_min = self.position[1] - self.width/2.
        self.ymax = self.position[1] + self.width/2.


class Road():

    """A road is a list of lanes, and a list of vehicles."""

    def __init__(self,
                 vehicles,
                 lanes,
                 bidirectional=False,
                 np_random: np.random.RandomState = None):
        """
        New road.

        Parameters
        ----------
        vehicles: the vehicles driving on the road
        lanes: the lanes driving on the road
        bidirectional: allowing bidirectional traffic
        np.random.RandomState np_random: a random number generator for vehicle behavior
        """
        self.bidirectional = bidirectional
        self.vehicles = vehicles or []
        self.lanes = lanes or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.speed = None

    def generate_lanes(self, lanes=4, length=50.):
        """
        Genarate and add lanes to the road.
        
        Parameters
        ----------
        lanes (int): the number of lanes to be genarates
        length (float): the length of the lane

        """
        for lane in range(lanes):
            origin = np.array([0, lane * Lane.DEFAULT_WIDTH + Lane.DEFAULT_LANE_SPACE])
            end = np.array([length, lane * Lane.DEFAULT_WIDTH + Lane.DEFAULT_LANE_SPACE])
            line_types = [LineType.CONTINUOUS if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS if lane == lanes - 1 else LineType.NONE]

            direction = self.np_random.choice([-1, 1]) if self.bidirectional else 1
            self.lanes.append(Lane(origin, end, line_types, direction=direction))

    def generate_random_vehicle(self, speed=None, lane_id=None, spacing=2.5, width = Vehicle.DEFAULT_WIDTH, length = Vehicle.DEFAULT_LENGTH):
        """
        Creates a random vehicle on the road.
        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.
        
        Parameters
        ----------
        speed (float): initial speed in [m/s]. If None, will be chosen randomly
        lane_id (int): id of the lane to spawn in
        spacing: safe distance to the front vehicle, 2.5 being the default
        """
        if speed is None:
            speed = self.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        if speed > Vehicle.DEFAULT_SPEEDS[1]:
            speed = Vehicle.DEFAULT_SPEEDS[1]
        if self.speed:
            if speed > self.speed:
                self.speed = speed
        else:
            self.speed = speed
        default_spacing = spacing*self.speed #Vehicle.DEFAULT_SPEEDS[1]
        id = lane_id if lane_id is not None else self.np_random.choice(len(self.lanes))
        lane = self.lanes[id]
        # do a quick search for the vehicles on that lane; could be more efficient
        lane_vehicles = []
        for v in self.vehicles:
            if v.lane == lane:
                lane_vehicles.append(v)

        pos_x = None
        pos_y = lane.start[1]
        if not len(lane_vehicles):
            pos_x = self.np_random.uniform(lane.start[0], lane.end[0])
        else:
            for _ in range(1000):  # give up after 1000 trials
                p = self.np_random.uniform(lane.start[0], lane.end[0])
                collision = False
                for v in lane_vehicles:
                    if abs(v.position[0] - p) <= (v.length + default_spacing):
                        collision = True
                        break
                if not collision:
                    pos_x = p
                    break
        if not pos_x:
            return
        v = Vehicle(np.array([pos_x, pos_y]), lane, speed, length, width)
        self.vehicles.append(v)

    def step(self, dt):
        """
        Update each vehicle on the road.
        """
        for v in self.vehicles:
            v.step(dt)

    def get_last_lane_Y(self):
        return self.lanes[-1].end[1]
    
    def get_first_lane_Y(self):
        return self.lanes[0].end[1]

    def __repr__(self):
        return self.vehicles.__repr__()
