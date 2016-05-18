""" Implement list at https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies """

class ExplorationStrategy(object):

    def __init__(self, env, args):
        self.env = env
        self.history_length = args.history_length
        self.exploration_train_strategy = args.exploration_train_strategy
        self.exploration_test_strategy = args.exploration_test_strategy
    

    def epsilon_decreasing_strategy(self, t):
        if t < self.history_length or random.random() < self.exploration_rate:
            return self.env.action_space.sample()
       else:
            return None
        
# consider adding https://github.com/asmith26/Vose-Alias-Method
