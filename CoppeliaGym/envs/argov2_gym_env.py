import gym
from gym.spaces import Discrete, Box

import numpy as np
import os

from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import JointType, JointMode
from pyrep.backend import sim, simConst

class ArgoV2Env(gym.Env):
    def __init__(self,
                 scene: str = 'ArgoV2.ttt',
                 dt: float = 0.05,
                 headless: bool = False) -> None:

        self._pr = PyRep()
        
        self.dt = dt

        scene_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'robots',
            scene)
        
        self._pr.launch(scene_path, headless=headless)
        self._pr.set_simulation_timestep(self.dt)

        if not headless:
            sim.simSetBoolParameter(simConst.sim_boolparam_browser_visible, False)
            sim.simSetBoolParameter(simConst.sim_boolparam_hierarchy_visible, False)
            sim.simSetBoolParameter(simConst.sim_boolparam_console_visible, False)
        #self._pr.start()

        # Custom env
        self.num_joints = 12
        self.joints = [Joint('Joint{}'.format(i)) for i in range(1, self.num_joints + 1)]
        # Collisions are defined in CoppeliaSim. For this simulation just links and body
        self.collisions = [sim.simGetCollisionHandle('Collision{}'.format(i)) for i in range(4, 17)]
        self.ref = Dummy('ArgoV2_reference')

        self.action_space = Box(low=np.array([-np.pi / 3] * self.num_joints), high=np.array([np.pi / 3] * self.num_joints), dtype=np.float32) #Discrete(self.num_joints)

        bl = np.array([-np.inf] * 3 + [-np.pi / 3] * self.num_joints + [-np.inf] * 6 + [-np.pi / 3] * self.num_joints)
        bh = np.array([np.inf] * 3 + [np.pi / 3] * self.num_joints + [np.inf] * 6 + [np.pi / 3] * self.num_joints)
        self.observation_space = Box(low=bl, high=bh, dtype=np.float32)

        self.target = np.array([1e3 / np.sqrt(2), -1e3 / np.sqrt(2)])
        self.progress = -np.linalg.norm(self.target - self.ref.get_position()[:2]) / self.dt

    def step(self, action):
        for key, joint in enumerate(self.joints):
            joint.set_joint_target_position(action[key])

        self._pr.step()
        cur_progress = -np.linalg.norm(self.target - self.ref.get_position()[:2]) / self.dt

        #velocity = np.linalg.norm(xy_pos_after - xy_pos_before) #/ dt
        #reward = velocity

        reward = cur_progress - self.progress
        self.progress = cur_progress

        done = False
        for part in self.collisions:
            if sim.simReadCollision(part):
                done = True
                break
        
        # expand
        state = np.concatenate((self.ref.get_position(),
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
                               [joint.get_joint_position() for joint in self.joints],
                               self.ref.get_velocity()[0],
                               self.ref.get_velocity()[1],
                               [joint.get_joint_velocity() for joint in self.joints]))
        self.state = state
        return state
    
    def close(self) -> None:
        self._pr.stop()
        self._pr.shutdown()
