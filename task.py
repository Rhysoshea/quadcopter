import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 1
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime

        '''
        try limiting the action space and state space to only vertical
        set the rotor speeds equal to each other if possible
        tweak the neural network in model.py
        '''

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def distance_to_target(self, current, target):
        import math
        d_1 = current[0] - target[0]
        d_2 = current[1] - target[1]
        d_3 = current[2] - target[2]
        deviation = math.sqrt((d_1)**2 + (d_2)**2 + (d_3)**2)
        return deviation

    def avg_rotor_speed(self, rotor_speeds):
        import math
        average = np.average(rotor_speeds)
        s_0 = rotor_speeds[0]-average
        s_1 = rotor_speeds[1]-average
        s_2 = rotor_speeds[2]-average
        s_3 = rotor_speeds[3]-average
        deviation = math.sqrt((s_0)**2 + (s_1)**2 + (s_2)**2 + (s_3)**2)
        return deviation


    def get_reward(self, rotor_speeds, state):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # distance_delta = self.distance_to_target(self.sim.pose, self.target_pos)
        # angular_v_delta = self.distance_to_target(self.sim.angular_v, [0,0,0])
        # speed_delta = self.avg_rotor_speed(rotor_speeds)

        ## runtime
        # if self.runtime - self.sim.time == 0:
        #     bonus = .5
        # else:
        #     bonus = 0
        # reward = 1.0 - (0.5*dist_rew) #+ (-0.2 * angular_v_delta) +  (-0.005 * speed_delta) # + bonus

        reward = (1/(abs(state[2] - self.target_pos[2])/10))
        # dist_rew = distance_delta
        if (abs(state[2] - self.target_pos[2])) <= 2:
            reward = 5
        ## penalise for crashing
        if self.sim.pose[2] == 0:
            reward = -10

        return reward

    def step(self, rotor_speeds, state):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds,state)
            pose_all.append(self.sim.pose)
        # new position
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
