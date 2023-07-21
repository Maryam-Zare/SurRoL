"""
Helper functions
"""
from typing import Tuple
import numpy as np
import pybullet as p
import cv2
from scipy import interpolate

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
RGB_COLOR_255 = [(230, 25, 75),  # red
                 (60, 180, 75),  # green
                 (255, 225, 25),  # yellow
                 (0, 130, 200),  # blue
                 (245, 130, 48),  # orange
                 (145, 30, 180),  # purple
                 (70, 240, 240),  # cyan
                 (240, 50, 230),  # magenta
                 (210, 245, 60),  # lime
                 (250, 190, 190),  # pink
                 (0, 128, 128),  # teal
                 (230, 190, 255),  # lavender
                 (170, 110, 40),  # brown
                 (255, 250, 200),  # beige
                 (128, 0, 0),  # maroon
                 (170, 255, 195),  # lavender
                 (128, 128, 0),  # olive
                 (255, 215, 180),  # apricot
                 (0, 0, 128),  # navy
                 (128, 128, 128),  # grey
                 (0, 0, 0),  # white
                 (255, 255, 255)]  # black


class Boundary(object):
    """ A boundary class to sample object in its work space. """

    def __init__(self, boundary: [list, tuple, np.ndarray]):
        self._boundary, self._area = None, 0
        self.set_boundary(boundary)
        self._contained_objects = []
        self._contained_object_positions = []

    def _get_position_within_boundary(self) -> tuple:
        x = np.random.uniform(
            self._boundary[0][0], self._boundary[0][1])
        y = np.random.uniform(
            self._boundary[1][0], self._boundary[1][1])
        z = np.random.uniform(
            self._boundary[2][0], self._boundary[2][1])
        return x, y, z

    def set_boundary(self, boundary: [list, tuple, np.ndarray]):
        assert len(boundary) == 3  # assume the boundary is a cube
        if not isinstance(boundary, np.ndarray):
            self._boundary = np.array(boundary)
        else:
            self._boundary = boundary
        self._area = float(np.prod(self._boundary[:, 1] - self._boundary[:, 0]))

    def get_area(self) -> float:
        return self._area

    def add(self, obj_id: int,
            sample: bool = True,
            min_rotation: tuple = (0.0, 0.0, -3.14),
            max_rotation: tuple = (0.0, 0.0, 3.14),
            min_distance: float = 0.01) -> bool:
        """ Returns true if can add and adds it or do not change the position (sample)
        assume the object is the Base object
        rotation_limits: how mush we allow it to rotate from its original position
        """
        if not sample:
            # simply add the object into self._contained_objects
            success = True
            pos, rotation = p.getBasePositionAndOrientation(obj_id)
            new_pos = np.array(pos)
        else:
            # sample the position and rotation within the boundary
            # Rotate the bounding box randomly
            rotation = np.random.uniform(list(min_rotation), list(max_rotation))
            rotation = p.getQuaternionFromEuler(rotation)

            success, attempt_num = False, 0
            new_pos = None
            while not success and attempt_num < 100:
                new_pos = np.array(self._get_position_within_boundary())
                success = True
                for obj in self._contained_object_positions:
                    if np.linalg.norm(new_pos - obj) < min_distance:
                        success = False
                        break
                attempt_num += 1
        if success:
            p.resetBasePositionAndOrientation(obj_id, new_pos, rotation)
            self._contained_objects.append(obj_id)
            self._contained_object_positions.append(new_pos)
        return success

    def clear(self) -> None:
        self._contained_objects = []


class Trajectory(object):
    """ Generate a 2-D (x, y) trajectory that follows a randomly placed square path with a smooth traversal. """

    def __init__(self, workspace_limits: np.ndarray, steps_between_corners=250, seed=1024):
        self.workspace_limits = workspace_limits.copy()
        self.steps_between_corners = steps_between_corners
        self._seed = seed
        self.xi, self.yi = None, None
        self._step = 0
        self.generate_trajectory()

    def generate_trajectory(self):
        """ Generate a smooth trajectory that follows a randomly placed square path. """
        st0 = np.random.get_state()
        if self.xi is None and self.yi is None:
            # Using the seed to generate a specific trajectory
            np.random.seed(self._seed)

        # Randomly place the square path within the workspace limits
        x_min, x_max = self.workspace_limits[0]
        y_min, y_max = self.workspace_limits[1]
        side_length = min(x_max - x_min, y_max - y_min) * 0.5  # Adjust the side length as needed
        x_center = np.random.uniform(x_min + side_length / 2, x_max - side_length / 2)
        y_center = np.random.uniform(y_min + side_length / 2, y_max - side_length / 2)

        # Define the four corners of the square
        corners = [
            [x_center - side_length / 2, y_center - side_length / 2],
            [x_center - side_length / 2, y_center + side_length / 2],
            [x_center + side_length / 2, y_center + side_length / 2],
            [x_center + side_length / 2, y_center - side_length / 2]
        ]

        # Generate a smooth path by evenly distributing the steps between each corner
        num_points = self.steps_between_corners * 4  # Adjust the number of points for a smoother curve
        xi, yi = [], []

        for i in range(4):
            curr_corner = corners[i]
            next_corner = corners[(i + 1) % 4]

            for j in range(self.steps_between_corners):
                t = j / float(self.steps_between_corners)
                x = (1 - t) * curr_corner[0] + t * next_corner[0]
                y = (1 - t) * curr_corner[1] + t * next_corner[1]
                xi.append(x)
                yi.append(y)

        self.xi, self.yi = xi, yi

        if self.xi is None and self.yi is None:
            # Restore the numpy state
            np.random.set_state(st0)

    def step(self) -> list:
        """ Return the next (x, y) position and move to the next corner after a specified number of steps. """
        self._step = (self._step + 1) % len(self.xi)
        x, y = self.xi[self._step], self.yi[self._step]

        return [x, y]

    # Rest of the class methods remain unchanged...


    def get_step(self) -> int:
        return self._step

    def set_step(self, step: int):
        self._step = step

    def reset(self):
        """ Reset the _step to be the origin. """
        self._step = 0

    def seed(self, seed: int):
        self._seed = seed


def get_centroid(mask: np.ndarray, target: int) -> Tuple[bool, np.ndarray]:
    """
    Get the centroid of the target
    :return: True with normalized centroids [-1, 1] if target in the mask picture else False
    """
    target_mask = (mask == target).astype(np.float)
    img_shape = mask.shape[:2]
    if np.sum(target_mask > 0):
        M = cv2.moments(target_mask)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        normalized_x = (cX - img_shape[1] / 2) / (img_shape[1] - img_shape[1] / 2)
        normalized_y = (cY - img_shape[0] / 2) / (img_shape[0] - img_shape[0] / 2)
        return True, np.array([normalized_x, normalized_y])
    else:
        return False, np.array([np.NaN, np.NaN])
