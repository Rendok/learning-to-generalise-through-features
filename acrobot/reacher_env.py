from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv, Camera
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np
from pybulletgym.envs.roboschool.robots.manipulators.reacher import Reacher
from gym.spaces import Box
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment

import pybullet as p


class ReacherBulletEnv(BaseBulletEnv, py_environment.PyEnvironment):
    def __init__(self,
                 encoding_net=None,
                 render=False,
                 obs_type="float",
                 max_time_step=40,
                 same_init_state=False):

        self.robot = Reacher()
        BaseBulletEnv.__init__(self, self.robot)

        self.scene = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.camera = Camera()
        self.isRender = render
        self._seed()
        self._cam_dist = 0.6
        self._cam_yaw = 0
        self._cam_pitch = -90 #-30
        self._render_width = 128
        self._render_height = 128
        self._obs_type = obs_type
        self._time_step = 0
        self._max_time_step = max_time_step
        self._encoding_net = encoding_net

        self._goal_img = None
        self._goal_mean = None
        self._goal_var = None
        self._same_init_state = same_init_state


        if self._obs_type == "float":
            self.observation_space = array_spec.BoundedArraySpec(
                shape=(128, 128, 3), dtype=np.float32, minimum=0, maximum=1,
                name='observation')
            # self.observation_space = Box(
            #     shape=(128, 128, 3), dtype=np.float32, low=0, high=1)
        elif self._obs_type == "uint":
            self.observation_space = array_spec.BoundedArraySpec(
                shape=(128, 128, 3), dtype=np.uint8, minimum=0, maximum=255,
                name='observation')
            # self.observation_space = Box(
            #     shape=(128, 128, 3), dtype=np.uint8, low=0, high=255)
        else:
            raise ValueError

        self.action_space = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')

        self.reset()
        self.make_goal()

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    @property
    def goal_img(self):
        return self._goal_img

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def reset(self):
        super().reset()

        if self._same_init_state:
            self.robot_configuration_reset(0, 0, 1, 1)
        else:
            self.robot_configuration_reset(0, 0, self.np_random.uniform(low=-3.14, high=3.14),
                                           self.np_random.uniform(low=-3.14, high=3.14))

        self._time_step = 0

        obs = self.get_observation()

        return ts.restart(obs)

    def _step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        electricity_cost = (
                -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
                - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.HUD(state, a, False)

        obs = self.get_observation()
        rew = self._reward()

        self._time_step += 1
        # , sum(self.rewards), False, {}
        if self._terminal():
            return ts.termination(obs, rew)
        else:
            return ts.transition(obs, reward=rew, discount=1.0)

    def _terminal(self):
        if self._time_step >= self._max_time_step:
            return True
        else:
            return False

    def _reward(self):
        observation = self.get_observation()
        z = self._encoding_net.encode(observation[np.newaxis, ...])

        distance = np.linalg.norm(z.numpy() - self._goal_mean.numpy())

        # import matplotlib.pyplot as plt
        # plt.imshow(self._encoding_net.decode(z)[0, ...])
        # plt.show()

        return 5 - distance

    def get_observation(self):
        if self._obs_type == "float":
            return self.render('rgb_array').astype(np.float32) / 255.
        elif self._obs_type == "uint":
            return self.render('rgb_array').astype(np.uint8)

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)

    def make_goal(self):

        self.robot_configuration_reset(0, 0, 1, 1)

        self._goal_img = self.get_observation()

        if self._encoding_net is not None:
            self._goal_mean, self._goal_var = self._encoding_net.infer(self._goal_img[np.newaxis, ...])

    def robot_configuration_reset(self, target_x, target_y, central_joint, elbow_joint):
        TARG_LIMIT = 0.27

        if TARG_LIMIT < target_x < -TARG_LIMIT or TARG_LIMIT < target_y < -TARG_LIMIT\
                or 3.14 < central_joint < -3.14 or 3.14 < elbow_joint < -3.14:

            raise ValueError

        self.robot.jdict["target_x"].reset_current_position(target_x, 0)
        self.robot.jdict["target_y"].reset_current_position(target_y, 0)
        self.robot.fingertip = self.robot.parts["fingertip"]
        self.robot.target = self.robot.parts["target"]
        self.robot.central_joint = self.robot.jdict["joint0"]
        self.robot.elbow_joint = self.robot.jdict["joint1"]
        self.robot.central_joint.reset_current_position(central_joint, 0)
        self.robot.elbow_joint.reset_current_position(elbow_joint, 0)
