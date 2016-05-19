""" See https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies for more details """

class ExplorationStrategy(object):

    def __init__(self, env, args):
        self.env = env
        self.args = args

    def _do_we_explore(self, epsilon):
        # i.e. epsilon = exploration_rate
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return None
            
    
    def play_random(self):
        return self.env.action_space.sample()

    def epsilon_greedy_strategy(self):
        exploration_rate = float(0.05)
        return self._do_we_explore(exploration_rate)

    def epsilon_decreasing_strategy(self, t):
        exploration_rate_start = float(1)           # Exploration rate at the beginning of decay.
        exploration_rate_end = float(0.1)           # Exploration rate at the end of decay.
        exploration_decay_steps = float(1000000)    # How many steps to decay the exploration rate.
        # calculate decaying exploration rate
        if t < exploration_decay_steps:
            exploration_rate = exploration_rate_start - t * (exploration_rate_start - exploration_rate_end) / exploration_decay_steps
        else:
            exploration_rate = exploration_rate_end

        return _do_we_explore(exploration_rate)
        
# consider adding https://github.com/asmith26/Vose-Alias-Method for bias exploration(??)
