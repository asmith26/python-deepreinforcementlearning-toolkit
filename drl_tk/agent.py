import random
import logging
import numpy as np
logger = logging.getLogger(__name__)
from state_buffer import StateBuffer

class GymAgent(object):
    def __init__(self, env=Breakout-v0, net, replay_memory, args):
        self.env = env
        self.net = net
        self.mem = replay_memory
        self.buf = StateBuffer(args)
        self.action_n = action_space.n
        self.history_length = args.history_length
        self.exploration_train_strategy = args.exploration_train_strategy
        self.exploration_test_strategy = args.exploration_test_strategy
        #self.train_frequency = args.train_frequency
        #self.train_repeat = args.train_repeat
 

    def _restartRandom(self):
        self.env.reset()
        # perform random number of dummy actions to produce more stochastic games
        for  in xrange(random.randint(self.history_length, self.random_starts) + 1):
        reward = self.env.action_space.sample()
        screen = self.env.render()
        assert not self.env.done, "done state occurred during random initialization"
        # add dummy states to buffer
        self.buf.add(screen)

    def act(self, exploration_rate):
        # FOR BASE AGENT, perhasp use: raise NotImplementedError
        callbacks.on_act_begin()
        logging.info(callbacks.params["observation"])
        # determine whether to explore
        action = self.exploration_strategy(t)
        if action:
          logger.debug("Explore action = %d" % action)
        else:
            qvalues = net.predict(memory)
            action = np.argmax(qvalues[0])

        
        # otherwise choose action with highest Q-value
        state = self.buf.getStateMinibatch()
        # for convenience getStateMinibatch() returns minibatch
        # where first item is the current state
        qvalues = self.net.predict(state)
        assert len(qvalues[0]) == self.num_actions
        # choose highest Q-value of first state
        action = np.argmax(qvalues[0])
        logger.debug("Predicted action = %d" % action)

        # perform the action
        reward = self.env.act(action)
        screen = self.env.getScreen()
        terminal = self.env.isTerminal()

        # print reward
        if reward <> 0:
          logger.debug("Reward: %d" % reward)

        # add screen to buffer
        self.buf.add(screen)

        # restart the game if over
        if terminal:
            logger.debug("Terminal state, restarting")
            self._restartRandom()

        # call callback to record statistics
        act_logs = {}
        act_logs['observation'] = 
        self.callback.on_act_end(act logs={observation, reward, done, info, exploration_rate})

         ##callbacks.on_act_end(epoch

        return observation, reward, done, info

  def play_random(self, random_steps):
    # play given number of steps
    for t in xrange(random_steps):
      # use exploration rate 1 = completely random
      self.act(1)

  def train(self, train_steps, epoch = 0):
    # do not do restart here, continue from testing
    #self._restartRandom()
    # play given number of steps
    for t in xrange(train_steps):
      # perform game step
      action, reward, screen, terminal = self.act(self._explorationRate(t))
      self.mem.add(action, reward, screen, terminal)
      # train after every train_frequency steps
      if self.mem.count > self.mem.batch_size and t % self.train_frequency == 0:
        # train for train_repeat times
        for j in xrange(self.train_repeat):
          # sample minibatch
          minibatch = self.mem.getMinibatch()
          # train the network
          self.net.train(minibatch, epoch)
      # increase number of training steps for epsilon decay
      self.total_train_steps += 1

  def test(self, test_steps, epoch = 0):
    # just make sure there is history_length screens to form a state
    self._restartRandom()
    # play given number of steps
    for t in xrange(test_steps):
      # perform game step
      self.step(self.exploration_rate_test)

  def play(self, num_games):
    # just make sure there is history_length screens to form a state
    self._restartRandom()
    for t in xrange(num_games):
      # play until terminal state
      terminal = False
      while not terminal:
        action, reward, screen, terminal = self.step(self.exploration_rate_test)
