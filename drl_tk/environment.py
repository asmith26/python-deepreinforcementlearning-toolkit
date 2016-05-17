#!/usr/bin/env python

""" Based on https://github.com/tambetm/simple_dqn/blob/master/src/environment.py """

import sys
import os
import logging
import cv2
logger = logging.getLogger(__name__)
import gym


class Environment:
  def __init__(self):
    pass

  def numActions(self):
    # Returns number of actions
    raise NotImplementedError

  def restart(self):
    # Restarts environment
    raise NotImplementedError

  def act(self, action):
    # Performs action and returns reward
    raise NotImplementedError

  def getScreen(self):
    # Gets current game screen
    raise NotImplementedError

  def isTerminal(self):
    # Returns if game is done
    raise NotImplementedError


class GymEnvironment(Environment):
   def __init__(self, env_id, args):
      self.gym = gym.make(env_id)
      self.obs = None
      self.terminal = None
      # OpenCV expects width as first and height as second s
      # GITHUBself.dims = (args.screen_width, args.screen_height)
 
   def numActions(self):
      # GITHUB assert isinstance(self.gym.action_space, gym.spaces.Discrete)
      return self.gym.action_space.n
 
   def restart(self):
      self.gym.reset()
      self.obs = None #COMMENTED GITHUB
      self.terminal = None
 
   def act(self, action):
      self.obs, reward, self.terminal, _ = self.gym.step(action)
      return reward
 
   def getScreen(self):
      assert self.obs is not None
      return self.obs
      # GITHUB return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), self.dims)
 
   def isTerminal(self):
      assert self.terminal is not None
      return self.terminal
