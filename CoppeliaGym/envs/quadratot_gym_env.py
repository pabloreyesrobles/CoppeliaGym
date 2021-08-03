import gym
from gym.spaces import Discrete, Box

import numpy as np
import os

from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import JointType, JointMode
from pyrep.backend import sim

class QuadratotEnv(gym.Env):
    def __init__(self,
                 scene: str = 'Quadratot.ttt',
                 dt: float = 0.05,
                 headless: bool = True) -> None:

        self._pr = PyRep()
        
        self.dt = dt

        scene_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'robots',
            scene)
        
        self._pr.launch(scene_path, headless=headless)
        self._pr.set_simulation_timestep(self.dt)
        #self._pr.start()

        # Custom env
        self.joints = [Joint('Joint{}'.format(i)) for i in range(1, 10)]
        self.collisions = [sim.simGetCollisionHandle('Collision{}'.format(i)) for i in range(4, 14)]
        self.ref = Dummy('Quadratot_reference')

        self.action_space = Box(low=np.array([-np.pi / 3] * 9), high=np.array([np.pi / 3] * 9), dtype=np.float32) #Discrete(9)

        bl = np.array([-np.inf] * 7 + [-np.pi / 3] * 9 + [-np.inf] * 6 + [-np.pi / 3] * 9)
        bh = np.array([np.inf] * 7 + [np.pi / 3] * 9 + [np.inf] * 6 + [np.pi / 3] * 9)
        self.observation_space = Box(low=bl, high=bh, dtype=np.float32)

    def step(self, action):
        for key, joint in enumerate(self.joints):
            joint.set_joint_target_position(action[key])

        xy_pos_before = self.ref.get_position()[:2]
        self._pr.step()
        xy_pos_after = self.ref.get_position()[:2]

        velocity = np.linalg.norm(xy_pos_after - xy_pos_before) / self.dt

        done = False
        for part in self.collisions:
            if sim.simReadCollision(part):
                done = True
                break
        
        reward = velocity

        # expand
        state = np.concatenate((self.ref.get_position(),
                               self.ref.get_quaternion(),
                               [joint.get_joint_position() for joint in self.joints],
                               self.ref.get_velocity()[0],
                               self.ref.get_velocity()[1],
                               [joint.get_joint_velocity() for joint in self.joints]))

        self.state = state

        return self.state, reward, done, {}

    def render(self):
        pass

    def reset(self):
        self._pr.stop()
        self._pr.start()

        state = np.concatenate((self.ref.get_position(),
                               self.ref.get_quaternion(),
                               [joint.get_joint_position() for joint in self.joints],
                               self.ref.get_velocity()[0],
                               self.ref.get_velocity()[1],
                               [joint.get_joint_velocity() for joint in self.joints]))
        self.state = state
        return state
    
    def close(self) -> None:
        self._pr.stop()
        self._pr.shutdown()
