import math
import utils
import numpy as np

class Env:
    def __init__(self, path: list[float, float, float], delta_t: int = 5, threshold: int = 10, max_step: int = 100):
        self.path = path
        self.delta_t = delta_t
        self.threshold = threshold
        self.max_step = max_step
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.state = np.array((path[0][0], path[0][1], path[0][2], 0, 0, 0), dtype=np.float32)  # (lat, lon, ele, heading, tilting, speed)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.state = np.array((self.path[0][0], self.path[0][1], self.path[0][2], 0, 0, 0), dtype=np.float32)  # (lat, lon, ele, heading, tilting, speed)
        return self.state
    
    def step(self, action):
        heading, tilting, speed = action
        lat, lon, ele = utils.get_next_position(self.state[0], self.state[1], self.state[2], heading, tilting, speed, self.delta_t)
        self.state = np.array((lat, lon, ele, heading, tilting, speed), dtype=np.float32)

        self.reward = -(utils.get_nearest_dist(lat, lon, ele, self.path)/50.0)
        self.reward -= 0.2 * abs(speed - 1.5)
        self.reward -= 0.05 * abs(self.state[3] - heading)
        self.reward -= 0.05 * abs(self.state[4] - tilting)

        goal_distance = utils.get_distance3D(lat, lon, ele, self.path[-1][0], self.path[-1][1], self.path[-1][2])
        if goal_distance < self.threshold:
            self.reward += 10
        
        self.current_step += 1
        if self.current_step >= self.max_step or utils.get_distance3D(self.state[0], self.state[1], self.state[2], self.path[-1][0], self.path[-1][1], self.path[-1][2]) < 10:
            self.done = True
        
        self.reward = np.clip(self.reward, -20, 20)

        return self.state, self.reward, self.done