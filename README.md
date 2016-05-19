# python-deepreinforcementlearning-toolkit

## Goals:
- Ability to rapidly prototype new reinforcement learning agents within the [OpenAI gym](https://github.com/openai/gym).
- Use gym terminology.
- Eay integration with Keras.

## Dependencies
- `env.monitor` (ran by `test_gym.py`) requires ffmpeg ([installation instructions](http://askubuntu.com/a/605210/450605))

## Acknowledgements
- Heavily based on https://github.com/tambetm/simple_dqn/ (particularly coding style) and [DeepMind's Atari Deep-Q-Learner](self.env.action_space.n)
- Many thanks to all!

## Agents

# Building a new agent

Required:
- `act(self, observation, reward)` 

# Existing agents
- `A3C.py` reproduces [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)
