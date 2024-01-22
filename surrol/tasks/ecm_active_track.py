import os
import time

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose, get_link_pose
)
from surrol.utils.robotics import (
    get_matrix_from_pose_2d,
    get_intrinsic_matrix,
)
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend

import matplotlib.pyplot as plt
plt.ion()

from surrol.utils.utils import RGB_COLOR_255, Boundary, Trajectory, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
import numpy as np
from scipy.spatial.transform import Rotation as R
import math



class ActiveTrack(EcmEnv):
    """
    Active track is not a GoalEnv since the environment changes internally.
    The reward is shaped.
    """
    ACTION_MODE = 'cVc'
    QPOS_ECM = (0, 0, 0.02, 0)
    WORKSPACE_LIMITS = ((-0.3, 0.6), (-0.4, 0.4), (0.05, 0.05))
    CUBE_NUMBER = 18

    def __init__(self, render_mode=None):
        # to control the step
        self._step = 0

        super(ActiveTrack, self).__init__(render_mode)
        self.camera_positions = []  # To store camera positions
        self.cube_positions = []
        self.ax = None
        self.fig = None
        self.camera_path = None  # Define camera_path as a class attribute
        self.cube_path = None  # Define cube_path as a class attribute


    def record_camera_position(self, position):
        self.camera_positions.append(position)

    def record_cube_position(self, position):
        self.cube_positions.append(position)

    # Implement a method to plot camera and cube paths
    def plot_paths(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.camera_path, = self.ax.plot([], [], label='Camera Path', linestyle='-', marker='o', markersize=7)
            self.cube_path, = self.ax.plot([], [], label='Red Cube Path', linestyle='-', marker='s', markersize=5)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_title('Camera and Red Cube Paths')
            self.ax.legend()
            self.ax.grid(True)
        # Extract x, y, and z coordinates from camera and cube positions
        camera_x, camera_y = zip(*self.camera_positions)
        cube_x, cube_y, cube_z = zip(*self.cube_positions)

        # Update the data for camera and cube paths
        self.camera_path.set_data(camera_x, camera_y)
        self.cube_path.set_data(cube_x, cube_y)
        x_min = min(camera_x + cube_x)
        x_max = max(camera_x + cube_x)
        y_min = min(camera_y + cube_y)
        y_max = max(camera_y + cube_y)
        self.ax.set_xlim(x_min - 0.1, x_max + 0.1)  # Adjust the margins as needed
        self.ax.set_ylim(y_min - 0.1, y_max + 0.1)
        # Update the plot
        self.fig.canvas.flush_events()
        plt.show()
        plt.pause(0.01)

        
        
    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        centroid = obs[-3: -1]
        if not (-1.2 < centroid[0] < 1.1 and -1.1 < centroid[1] < 1.1):
            # early stop if out of view
            done = True
        info['achieved_goal'] = centroid
        self.obs = obs
        
        #self.plot_paths()
        

        
        return obs, reward, done, info

    def plott(self):
        self.camera_positions = []  # To store camera positions
        self.cube_positions = []
        plt.close()
        self.fig = None
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ Dense reward."""
        centroid, wz = achieved_goal[-3: -1], achieved_goal[-1]
        d = goal_distance(centroid, desired_goal) / 2
        reward = 1 - (d + np.linalg.norm(wz) * 0.1)  # maximum reward is 1, important for baseline DDPG
        return reward

    def _env_setup(self):
        super(ActiveTrack, self)._env_setup()
        self.use_camera = True

        # robot
        self.ecm.reset_joint(self.QPOS_ECM)

        # trajectory
        traj = Trajectory(self.workspace_limits, seed=None)
        self.traj = traj
        self.traj.set_step(self._step)

        # target cube
        b = Boundary(self.workspace_limits)
        x, y = self.traj.step()
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                            (x * self.SCALING, y * self.SCALING, 0.05 * self.SCALING),
                            p.getQuaternionFromEuler(np.random.uniform(np.deg2rad([0, 0, -90]),
                                                                       np.deg2rad([0, 0, 90]))),
                            globalScaling=0.8 * self.SCALING)
        color = RGB_COLOR_255[0]
        p.changeVisualShape(obj_id, -1,
                            rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                            specularColor=(0.1, 0.1, 0.1))
        self.obj_ids['fixed'].append(obj_id)  # 0 (target)
        self.obj_id = obj_id
        b.add(obj_id, sample=False, min_distance=0.12)
        self._cid = p.createConstraint(obj_id, -1, -1, -1,
                                       p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [x, y, 0.05 * self.SCALING])

        # other cubes
        b.set_boundary(self.workspace_limits + np.array([[-0.2, 0], [0, 0], [0, 0]]))
        for i in range(self.CUBE_NUMBER):
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                                (0, 0, 0.05), (0, 0, 0, 1),
                                globalScaling=0.8 * self.SCALING)
            color = RGB_COLOR_255[1 + i // 2]
            p.changeVisualShape(obj_id, -1,
                                rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                                specularColor=(0.1, 0.1, 0.1))
            # p.changeDynamics(obj_id, -1, mass=0.01)
            b.add(obj_id, min_distance=0.12)

    def _get_obs(self) -> np.ndarray:
        robot_state = self._get_robot_state()
        # may need to modify
        _, mask = self.ecm.render_image()
        in_view, centroids = get_centroid(mask, self.obj_id)

        if not in_view:
            # out of view; differ when the object is on the boundary.
            pos, _ = get_body_pose(self.obj_id)
            centroids = self.ecm.get_centroid_proj(pos)
            print(" -> Out of view! {}".format(np.round(centroids, 4)))

        observation = np.concatenate([
            robot_state, np.array(in_view).astype(np.float).ravel(),
            centroids.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
        ])
        return observation

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0.])
        return goal.copy()
    
    
    import numpy as np

    def get_camera_intersection_2d(self):
        """
        Calculate the 2D coordinates on the workspace plane where the camera is currently pointing.

        
        """
        pos, orn = get_link_pose(self.ecm.body, self.ecm.TIP_LINK_INDEX)
        tip_world = get_matrix_from_pose_2d((pos, orn))
        tip_world[:3, 3] /= self.SCALING
        # Extract camera axis and position from the tip_world matrix
        cam_z = tip_world[0: 3, 2]
        cam_pos = tip_world[0: 3, 3]
        plane_z = np.array([0., 0., 1])
        z0 = 0.05
        plane_pos = np.array([0, 0, z0])
        t = ((plane_pos[0] - cam_pos[0]) * plane_z[0] +
             (plane_pos[1] - cam_pos[1]) * plane_z[1] +
             (plane_pos[2] - cam_pos[2]) * plane_z[2]) \
            / (plane_z[0] * cam_z[0] + plane_z[1] * cam_z[1] + plane_z[2] * cam_z[2])

        # Calculate the intersection point (intersection in 3D)
        intersection_3d = cam_pos + t * cam_z

        # Extract the x and y coordinates of the intersection point (2D on workspace plane)
        intersection_2d = intersection_3d[:2]

        return intersection_2d


        
 
# Function to get the position of a cube (assuming you know its body ID)
    def get_cube_position(self, cube_body_id):
        cube_position, _ = p.getBasePositionAndOrientation(cube_body_id)
        return cube_position

   

    def _step_callback(self):
        """ Move the target along the trajectory
        """
        for _ in range(10):
            x, y = self.traj.step()
            self._step = self.traj.get_step()
            pivot = [x, y, 0.05 * self.SCALING]
            p.changeConstraint(self._cid, pivot, maxForce=50)
            p.stepSimulation()
            
            #camera_matrix = self.get_camera_matrix()  # Obtain the camera transformation matrix (4x4) here
            #workspace_matrix = self.get_workspace_matrix()  # Obtain the workspace transformation matrix (4x4) here

            # Call the get_camera_center_2d function to get the 2D coordinate in workspace space
            camera_position = self.get_camera_intersection_2d()

            
            cube_position = self.get_cube_position(self.obj_id)      # Implement this method
            self.record_camera_position(camera_position)
            self.record_cube_position(cube_position)
            


    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        centroid = obs[-3: -1]
        cam_u = centroid[0] * RENDER_WIDTH
        cam_v = centroid[1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 8 and np.linalg.norm(self.ecm.wz) < 0.1:
            # e difference is small enough
            action = np.zeros(3)
        else:
            print("Pixel error: {:.4f}".format(np.linalg.norm(self.ecm.homo_delta)))
            # controller
            fov = np.deg2rad(FoV)
            fx = (RENDER_WIDTH / 2) / np.tan(fov / 2)
            fy = (RENDER_HEIGHT / 2) / np.tan(fov / 2)  # TODO: not sure
            cz = 1.0
            Lmatrix = np.array([[-fx / cz, 0., cam_u / cz],
                                [0., -fy / cz, cam_v / cz]])
            action = 0.5 * np.dot(np.linalg.pinv(Lmatrix), self.ecm.homo_delta).flatten() / 0.01
            if np.abs(action).max() > 1:
                action /= np.abs(action).max()
        return action


if __name__ == "__main__":
    env = ActiveTrack(render_mode='human')  # create one process and corresponding env

    env.test(horizon=200)
    env.close()
    time.sleep(2)
