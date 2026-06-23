# RL-ET-DeePC-for-soft-robotic-arm-control

<p align="center">
  <b>Reinforcement Learning-Enhanced Event-Triggered Data-Enabled Predictive Control for a 3D Cable-Driven Soft Robotic Arm</b>
</p>

<p align="center">
  <img src="assets/framework.png" width="850">
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#results">Results</a> |
  <a href="#hardware-setup">Hardware</a> |
  <a href="#citation">Citation</a>
</p>

## Overview

This repository provides the official implementation of RL-ET-DeePC, a reinforcement-learning-based event-triggered data-enabled predictive control framework for a three-dimensional cable-driven soft robotic arm.

Conventional DeePC requires solving a constrained optimization problem at every sampling instant. RL-ET-DeePC learns a state-dependent triggering policy that determines whether to compute a new optimal control sequence or reuse inputs from the previously computed prediction buffer.

The proposed framework achieves:

- up to 66% fewer optimization calls in simulation;
- approximately 34% fewer optimization calls on hardware;
- tracking accuracy comparable to periodic DeePC;
- more adaptive triggering behavior than static threshold-based methods.

## Method

At each control step, the reinforcement-learning agent observes the tracking error and its temporal variation. The agent selects one of two actions:

- `trigger`: solve the SVD-DeePC optimization problem;
- `skip`: reuse the next control input from the prediction buffer.

A supervisory mechanism forces re-optimization whenever the buffer is empty or the prediction horizon is exhausted.

## Supported Algorithms

- Proximal Policy Optimization
- Deep Q-Network
- Advantage Actor-Critic
- Periodic DeePC baseline
- Threshold-based event-triggered DeePC baseline
