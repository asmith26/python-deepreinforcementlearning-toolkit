import random
import logging
import numpy as np
logger = logging.getLogger(__name__)
from state_buffer import StateBuffer

class GymAgent(object):
    def __init__(self, env=Breakout-v0, net, replay_memory, exploration_strategy, args):
        self.env = env
        self.net = net
        self.mem = replay_memory
        self.exporation_strategy = exporation_strategy
        self.buf = StateBuffer(args)
        self.history_length = args.history_length
        #self.exploration_train_strategy = exploration_strategy.args.exploration_train_strategy
        #self.exploration_test_strategy = exploration_strategy.args.exploration_test_strategy
        self.train_net_frequency = args.train_net_frequency
        self.train_net_repeat = args.train_net_repeat
 

    def _restart_random(self):
        self.env.reset()
        # perform random number of dummy actions to produce more stochastic games
        for t in xrange(random.randint(self.history_length, self.random_starts) + 1):
            self.mem.action = self.env.action_space.sample()
            self.mem.observation, self.mem.reward, self.mem.done, self.mem.info = self.env.step(self.mem.action)
            assert not self.env.done, "done state occurred during random initialization"
            
            # add dummy states to buffer
#to be merged in replay_memor=self.mem here   self.buf.add(observation)

    def act(self, exploration_strategy):
# FOR BASE AGENT, perhasp use: raise NotImplementedError
        callbacks.on_act_begin()
        # determine whether to explore
        action = exploration_strategy()
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
        # perform the action, and update replay_memory
        self.mem.action = action
        self.mem.observation, self.mem.reward, self.mem.done, self.mem.info = self.env.step(self.mem.action)
        # add screen to buffer
        #self.buf.add(observation)
        # restart the game if over
        if done:
            self._restart_random()
        # call callback to log progress
        #MOVE THIS TO CALLBACK SELF.AGENT (need to add self stuff above - NO! USE e.g. buf.observations[last (obvisously replace with the actual number)]):
##        act_logs = {}
##        act_logs['observation'] = observation
##        act_logs['done'] = done
##        act_logs['reward'] = reward
##        act_logs['t'] = t
        self.callback.on_act_end(act)
#see statistics vs monitor
        return action, observation, reward, done, info

    def train(self, train_steps, episode = 0):
#CHECK WHY, INPARTICULAR SURELY WE DON'T NECCESSARILY HAVE 4STATES FOR CONVNET???        # do not do restart here, continue from testing
        #self._restart_random()
        # play given number of steps
        for t in xrange(train_steps):
            # update agent replay memory regarding t
            self.mem.t = t
            # perform game step
            self.act(self.exploration_train_strategy)
            # train after every train_frequency steps
            if self.mem.count > self.mem.batch_size and t % self.train_frequency == 0:
                # train for train_repeat times
                for j in xrange(self.train_net_repeat):
                    # sample minibatch
                    minibatch = self.mem.getMinibatch()
                    # train the network
                    self.net.train(minibatch, episode)
            # restart the game if over
            if self.mem.done:
                # just make sure there is history_length screens to form a state
                # perform random number of dummy actions to produce more stochastic games
                if t < random.randint(self.history_length, self.random_starts) + 1:
                    self.act(self.exploration_strategy.play_random)                

    def test(self, test_steps, episode = 0):      
        # play given number of steps
        for t in xrange(test_steps):
            # update agent replay memory regarding t
            # check if we trained
            if t == 0:
                test_start_t = self.mem.t
                # reset environment
                self.env.reset()
            self.mem.t = test_start_t + t
            # just make sure there is history_length screens to form a state
            # perform random number of dummy actions to produce more stochastic games
            if t < random.randint(self.history_length, self.random_starts) + 1:
                self.act(self.exploration_strategy.play_random)
            # perform game step
            self.act(self.exploration_test_strategy)

    def play(self, num_games):
        for t in xrange(num_games):
            # just make sure there is history_length screens to form a state
            # perform random number of dummy actions to produce more stochastic games
            if t < random.randint(self.history_length, self.random_starts) + 1:
                self.act(self.exploration_strategy.play_random)
            # play until terminal state
            while not self.mem.done:
                self.act(t, self.exploration_test_strategy)
