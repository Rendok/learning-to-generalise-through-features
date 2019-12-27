from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv, Camera
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np
from pybulletgym.envs.roboschool.robots.manipulators.reacher import Reacher
from gym.spaces import Box
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from models.vae_env_model import VAE

from tensorflow import norm

import pybullet as p


class ReacherBulletEnv(BaseBulletEnv, py_environment.PyEnvironment):
    def __init__(self,
                 mode,
                 # encoding_net=None,
                 render=False,
                 obs_type="float",
                 max_time_step=40,
                 obs_as_vector=False,
                 train_env="gym",
                 delta=1):

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
        self._time_step = 0
        self._max_time_step = max_time_step
        self._encoding_net = self.load_encoding_net()

        self._goal_img = None
        self._goal_mean = None
        self._goal_var = None
        self._obs_type = obs_type
        self._obs_as_vector = obs_as_vector
        self._train_env = train_env
        self._delta = delta
        self._mode = mode
        self._info_dict = {}

        if self._mode not in ["same_init_state", "delta", "rand_init_state"]:
            raise ValueError

        if self._train_env not in ["tf_agent", "gym"]:
            raise ValueError

        if self._obs_type == "float":
            if self._train_env == "tf_agent":
                self.observation_space = array_spec.BoundedArraySpec(
                    shape=(128, 128, 3), dtype=np.float32, minimum=0, maximum=1,
                    name='observation')
            elif self._train_env == "gym":
                if self._obs_as_vector:
                    self.observation_space = Box(
                        shape=(self._encoding_net.latent_dim,), dtype=np.float32, low=-50, high=50)
                else:
                    self.observation_space = Box(
                        shape=(128, 128, 3), dtype=np.float32, low=0, high=1)

        elif self._obs_type == "uint":
            if self._train_env == "tf_agent":
                self.observation_space = array_spec.BoundedArraySpec(
                    shape=(128, 128, 3), dtype=np.uint8, minimum=0, maximum=255,
                    name='observation')
            elif self._train_env == "gym":
                self.observation_space = Box(
                    shape=(128, 128, 3), dtype=np.uint8, low=0, high=255)

        else:
            raise ValueError

        if self._train_env == "tf_agent":
            self.action_space = array_spec.BoundedArraySpec(
                shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')

        elif self._train_env == "gym":
            self.action_space = Box(
                shape=(2,), dtype=np.float32, low=-1, high=1)

        super().reset()
        self.make_goal()
        self.reset()

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

        self._info_dict = {}

        if self._mode == "same_init_state":                         # 0, 0: works
            self.robot_configuration_reset(0, 0, 0, -1)  # 0, -1: works # -1, -1: doesn't
        elif self._mode == "rand_init_state":
            self.robot_configuration_reset(0, 0, self.np_random.uniform(low=-3.14, high=3.14),
                                           self.np_random.uniform(low=-3.14, high=3.14))
        elif self._mode == "delta":
            # goal coordinates
            a = 1
            b = 1
            self.robot_configuration_reset(0, 0, a + self._delta[0] * self.np_random.uniform(low=-3.14, high=3.14),
                                           b + self._delta[1] * self.np_random.uniform(low=-3.14, high=3.14))
        else:
            raise ValueError

        self._time_step = 0

        obs = self.get_observation(as_vector=self._obs_as_vector)

        self._info_dict['init_dist'] = self._calculate_distance_for_dict()

        if self._train_env == "tf_agent":
            return ts.restart(obs)
        elif self._train_env == "gym":
            return obs

    def _step(self, a):
        assert (not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        obs = self.get_observation(as_vector=self._obs_as_vector)
        rew = self._reward()

        self._time_step += 1
        # , sum(self.rewards), False, {}

        if self._train_env == "tf_agent":
            if self._terminal():
                return ts.termination(obs, rew)
            else:
                return ts.transition(obs, reward=rew, discount=1.0)

        elif self._train_env == "gym":
            if self._terminal():
                self._info_dict['final_dist'] = self._calculate_distance_for_dict()
                return obs, rew, True, self._info_dict
            else:
                return obs, rew, False, {}

    def _terminal(self):
        if self._time_step >= self._max_time_step:
            return True
        else:
            return False

    def _reward(self):

        observation = self.get_observation(as_vector=False)
        z = self._encoding_net.encode(observation[np.newaxis, ...])

        distance = norm(z - self._goal_mean)

        # import matplotlib.pyplot as plt
        # plt.imshow(self._encoding_net.decode(z)[0, ...])
        # plt.show()

        return 2 - distance.numpy()

    def get_observation(self, as_vector=False):
        assert isinstance(as_vector, bool)

        if as_vector:
            observation = self.get_observation(as_vector=False)
            return self._encoding_net.encode(observation[np.newaxis, ...])[0, ...]
        else:
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

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.robot.central_joint.set_velocity(5 * float(np.clip(a[0], -1, +1)))
        self.robot.elbow_joint.set_velocity(5 * float(np.clip(a[1], -1, +1)))

    @staticmethod
    def load_encoding_net(latent_dim=256):
        encoding_net = VAE(latent_dim, channels=3)

        try:
            weights_path = '/tmp/weights'
            encoding_net.load_weights(['en', 'de'], weights_path)
            return encoding_net
        except:
            pass

        try:
            weights_path = '/Users/dgrebenyuk/Research/dataset/weights'
            encoding_net.load_weights(['en', 'de'], weights_path)
            return encoding_net
        except:
            pass

        raise NotImplementedError

    def _calculate_distance_for_dict(self):
        observation = self.get_observation(as_vector=False)
        mu, _ = self._encoding_net.infer(observation[np.newaxis, ...])
        return norm(mu - self._goal_mean).numpy()
