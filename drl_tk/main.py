#!/usr/bin/env python

""" Based on https://github.com/tambetm/simple_dqn/blob/master/src/main.py """

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

from environment import ALEEnvironment, GymEnvironment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
from agent import Agent
from statistics import Statistics
import random
import argparse
import sys

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("env", default="Breakout-v0", help="Specfigy env id (e.g. Breakout-v0).")
envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")
#envarg.add_argument("--sound", type=str2bool, default=False, help="Play (or record) sound.")
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--repeat_action_probability", type=float, default=0, help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
envarg.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True, help="Use minimal action set.")
envarg.add_argument("--color_averaging", type=str2bool, default=True, help="Perform color averaging with previous frame.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--record_screen_path", help="Record game screens under this path. Subfolder for each game is created.")
envarg.add_argument("--record_sound_filename", help="Record game sound in this file.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")

#netarg.add_argument("--rescale_r", type=str2bool, help="Rescale rewards.")
#missing: bufferSize=512,valid_size=500,min_reward=-1,max_reward=1

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

antarg = parser.add_argument_group('Agent')
#antarg.add_argument("--exploration_train_strategy", type=float, default=1, help="Exploration rate at the beginning of decay.")
#antarg.add_argument("--exploration_test_strategy", type=float, default=, help="Exploration rate at the end of decay.")
#antarg.add_argument("--exploration_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
#antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
# ADD IF STATEMENT BELOW REGARDING SCHEDULE , AND ADD TO args.exploration_rate_train_schedule args.exploration_rate_test_schedule VARIABLE BASED ON i_episode (C.F. KERAS)
# ADD exploration type
antarg.add_argument("--train_net_frequency", type=int, default=4, help="Perform training after this many game steps.")
antarg.add_argument("--train_net_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

nvisarg = parser.add_argument_group('Visualization')
nvisarg.add_argument("--visualization_filters", type=int, default=4, help="Number of filters to visualize from each convolutional layer.")
nvisarg.add_argument("--visualization_file", help="Write layer visualization to this file.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per episode.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each episode.")
mainarg.add_argument("--num_episodes", type=int, default=200, help="How many episodes to run.")
mainarg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")
mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. episode and extension will be appended.")
mainarg.add_argument("--csv_file", help="Write training progress to this file.")
mainarg.add_argument("--env_monitor_dir", default="/tmp/cartpole-experiment-1" help="Where to write gym monitor logs.")
mainarg.add_argument("--env_monitor_dir_overwrite", type=str2bool, default=False, help="Force overwrite monitor logs at --env_monitor_dir.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_level)

if args.random_seed:
  random.seed(args.random_seed)

# INSTANTIATE CLASSES
#logger.handlers.pop()
env = gym.make(args.env)
mem = ReplayMemory(args.replay_size, args)
net = DeepQNetwork(env.numActions(), args)
exploration_strategy = ExplorationStrategy(args)
if args.load_weights:
    logger.info("Loading weights from %s" % args.load_weights)
    net.load_weights(args.load_weights)
agent = GymAgent(env, net, memory, exploration_strategy, args)
stats = Statistics(agent, net, mem, env, args)

callbacks._set_agent(agent)
#callbacks._set_params({})

if args.play_games:
  logger.info("Playing for %d game(s)" % args.play_games)
  stats.reset()
  # reset environment
  agent.env.reset() # possibly not needed
  agent.play(args.play_games)
  stats.write(0, "play")
  sys.exit("Game play completed")

if args.random_steps:
  # populate replay memory with random steps
  logger.info("Populating replay memory with %d random moves" % args.random_steps)
  stats.reset()
  agent.play_random(args.random_steps)
  stats.write(0, "random")

# loop over all num_episodes
callbacks.on_allepisodes_begin()
for i_episode in xrange(args.num_episodes):
    logger.info("i_episode #%d" % (i_episode + 1))
    
    callbacks.on_episode_begin(i_episode)

    if args.train_steps:
        logger.info(" Training for %d steps" % args.train_steps)
        stats.reset()
        agent.train(args.train_steps, i_episode)
        stats.write(i_episode + 1, "train")

        if args.save_weights_prefix:
            filename = args.save_weights_prefix + "_%d.prm" % (i_episode + 1)
            logger.info("Saving weights to %s" % filename)
            net.save_weights(filename)

    if args.test_steps:
        logger.info(" Testing for %d steps" % args.test_steps)
        stats.reset()
        agent.test(args.test_steps, i_episode)
        stats.write(i_episode + 1, "test")

    callbacks.on_episode_end(i_episode)

callbacks.on_allepisodes_end() # PERHAPS ADD STATS
stats.close()
logger.info("All done")
