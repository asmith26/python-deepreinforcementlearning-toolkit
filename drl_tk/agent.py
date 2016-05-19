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
        self.history_length = args.history_length
        self.exploration_train_strategy = args.exploration_train_strategy
        self.exploration_test_strategy = args.exploration_test_strategy
        self.train_net_frequency = args.train_net_frequency
        self.train_net_repeat = args.train_net_repeat
 

    def _restart_random(self):
        self.env.reset()
        # perform random number of dummy actions to produce more stochastic games
        for t in xrange(random.randint(self.history_length, self.random_starts) + 1):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            assert not self.env.done, "done state occurred during random initialization"
            # add dummy states to buffer
            self.buf.add(observation)

    def act(self, t, exploration_strategy):
        # FOR BASE AGENT, perhasp use: raise NotImplementedError
        callbacks.on_act_begin()
        logging.info(callbacks.params["observation"])
        # determine whether to explore
        action = exploration_strategy(t)
        if action:
          logger.debug("Explore action = {}".format(action))
        else:
            # otherwise choose action with highest Q-value
            state = self.buf.getStateMinibatch()
            # for convenience getStateMinibatch() returns minibatch
            # where first item is the current state
            qvalues = self.net.predict(state)
            assert len(qvalues[0]) == self.env.action_space.n
            # choose highest Q-value of first state
            action = np.argmax(qvalues[0])
            logger.debug("Predicted action = {}".format(action))
        # perform the action
        observation, reward, done, info = self.env.step(action)
        # add screen to buffer
        self.buf.add(observation)
        # restart the game if over
        if done:
            self._restart_random()
        # call callback to log progress
        act_logs = {}
        act_logs['observation'] = observation
        act_logs['done'] = done
        act_logs['reward'] = reward
        act_logs['t'] = t
        self.callback.on_act_end(act logs=act_logs)
#see statistics vs monitor
        return action, observation, reward, done, info

    def train(self, train_steps, episode = 0):
        # do not do restart here, continue from testing
        #self._restart_random()
        # play given number of steps
        for t in xrange(train_steps):
            # perform game step
            action, observation, reward, done, info = self.act(t, self.exploration_train_strategy)
#CHECK            self.mem.add(action, observation, reward, done, info)
            # train after every train_frequency steps
            if self.mem.count > self.mem.batch_size and t % self.train_frequency == 0:
                # train for train_repeat times
                for j in xrange(self.train_net_repeat):
                    # sample minibatch
                    minibatch = self.mem.getMinibatch()
                    # train the network
                    self.net.train(minibatch, episode)

    def test(self, test_steps, episode = 0):
        # just make sure there is history_length screens to form a state
        self._restart_random()
        # play given number of steps
        for t in xrange(test_steps):
            # perform game step
            self.act(t, self.exploration_test_strategy)

    def play(self, num_games):
        # just make sure there is history_length screens to form a state
        self._restart_random()
        for t in xrange(num_games):
            # play until terminal state
            done = False
            while not done:
                action, observation, reward, done, info = self.act(t, self.exploration_test_strategy)
                # add experiences to replay memory for visualization
                self.mem.add(action, observation, reward, done, info)
