# Lunar Lander Problem
This repository contains an implementation of a Deep Q-Network (DQN) to solve the Lunar Lander.

## Project-Overview
The Lunar Lander problem is a reinforcement learning task where an agent must learn to safely land a lunar module on a moon. 
This project uses a Deep Q-Network, a popular deep reinforcement learning algorithm, to train an agent to successfully complete this task.

## Problem Description
### Environment
The environment consist on the earth, moon and shuttle positions.
The movement is allowed only to the shuttle which updates its coordinates and the speeds based on the Euler Equations after its actions.
It is implemented also the gravitational field to the shuttle from the earth and the moon.

### Agent
The shuttle choose an action in each frame. The actions are:
- 0: Do Nothing
- 1: Turn Left
- 2: Go Straight
- 3: Turn Right
With which the shuttle actives the relative engines.

### State
The state is composed by 7 values:
- The x shuttle's coordinate
- The y shuttle's coordinate
- The angle of the shuttle
- The x shuttle's speed
- The y shuttle's speed
- The angular speed
- The Fuel

### Rewards
The Positive Rewards are:
- Incremental value based on the distance from the flag
- Incremental value based on the small speed
- Small reward if the shuttle angle is similar to the flag one
- Very large reward if the shuttle lands

The Negative Rewards are:
- Very large penalty if the shuttle crashes near the earth
- Large penalty if the shuttle crashes near to the moon
- The reward is zero if the shuttle angular speed is too high

### Landing Conditions
If the distance of the shuttle from the flag is small, the angle is similar and the speeds are close to zero the landing condition is met.

## Contents
- [src/Agent.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/Agent.py): File containing the Agent class
- [src/Environment.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/Environment.py): File containing the Environment Class
- [src/dynamics.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/dynamics.py): File containing the functions to update the shuttle's coordinates and speed
- [src/utils.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/utils.py): File containing additional functions, mostly for the state rendering
- [src/DQN.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/DQN.py): File containing the Class for the Deep Q-Network Model
- [src/actor_critic.py](https://github.com/Nefasto1/Reinforcement-Learning-Project/blob/main/src/actor_critic.py): File containing the Class for the Actor Critic Model

