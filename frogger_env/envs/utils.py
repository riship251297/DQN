import numpy as np


class Line:
    def __init__(self, point1, point2):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)

    def __repr__(self):
        return "({:.2f}, {:.2f}) --> ({:.2f}, {:.2f})".format(self.point1[0],
                                                              self.point1[1],
                                                              self.point2[0],
                                                              self.point2[1])

    def nearest_to_point(self, position):
        position = np.array(position)
        length = np.linalg.norm(self.point1 - self.point2, 2)
        dir = (self.point2 - self.point1)/length
        d = np.dot(dir, position - self.point1)/np.dot(dir,  dir)
        p = self.point1 + d * dir
        length1 = np.linalg.norm(self.point1 - position, 2)
        length2 = np.linalg.norm(self.point2 - position, 2)
        if (length1 + length2) > length:
            if length1 < length2:
                p = self.point1
            else:
                p = self.point2
        dist = np.linalg.norm(p - position, 2)
        return p, dist

    def ray_intersection(self, ray_origin, angle):
        ray_origin = np.array(ray_origin)
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        v1 = ray_origin - self.point1
        v2 = self.point2 - self.point1
        perp = np.array([-ray_dir[1], ray_dir[0]])
        dvP = np.dot(v2, perp)
        if dvP != 0:
            t1 = np.cross(v2, v1)/np.dot(v2, perp)
            t2 = np.dot(v1, perp)/np.dot(v2, perp)
            if t1 >= 0 and t2 >= 0 and t2 <= 1:
                return ray_origin + t1 * ray_dir, t1
        return [], np.inf


class Rectangle:
    def __init__(self, x_max, x_min, y_max, y_min):
        self.lines = []
        self.lines.append(Line([x_max, y_max], [x_max, y_min]))
        self.lines.append(Line([x_max, y_min], [x_min, y_min]))
        self.lines.append(Line([x_min, y_min], [x_min, y_max]))
        self.lines.append(Line([x_min, y_max], [x_max, y_max]))
        self.width = y_max - y_min
        self.length = x_max - x_min
        self.x = x_min
        self.y = y_min

    def nearest_to_point(self, position):
        min_dist = np.inf
        nearest_point = None
        for line in self.lines:
            point, dist = line.nearest_to_point(position)
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
        return nearest_point, min_dist

    def ray_intersection(self, ray_origin, angle):
        min_dist = np.inf
        nearest_point = None
        for line in self.lines:
            point, dist = line.ray_intersection(ray_origin, angle)
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
        return nearest_point, min_dist

    def rectangle_intersection(self, rect):
        # print(self.x, self.y, self.length, self.width)
        # print(rect.x, rect.y, rect.length, rect.width)
        if (rect.x < self.x + self.length) and \
           (rect.x + rect.length > self.x) and \
           (rect.y < self.y + self.width) and \
           (rect.y + rect.width > self.y):
            return True
        return False
