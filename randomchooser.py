import numpy as np

class Chooser():
    def __init__(self, space, probs=None, duration= 30):
        self.space = space
        self.duration = -1
        self.weights = probs
        if probs is not None and space != len(probs):
            raise Exception("Action space and probability must be the same dimension!")
        if probs is not None:
            self.action_space = np.arange(space)
            self.pool_sizes = np.arange(space)+1
            # The following vars only matter for a variably chooser
            # These are soft constant params
            self.probs = np.array(probs) # this is a probabilty distribution that will be used to decide how many
                                         # actions will be available to choose from at a given time. Ranging from 1 - space.
            self.probs = self.probs / self.probs.sum()
            self.duration = duration

            # These are volatile/dynamic params that may change randomly over time
            self.clock = duration # acts as a timer for when to randomize the move pool and hist
            self.move_pool = np.arange(space)
            self.hist = np.ones(space)*10
            self.reset_hist()
        else:
            self.probs = None

    def get_random_action(self):
        if self.probs is None:
            return self.get_uniformly_random_action()
        else:
            return self.get_variably_random_action()

    def get_uniformly_random_action(self):
        return np.random.choice(self.space)


    def get_variably_random_action(self):
        # pull a move from the current move pool with the current probability distribution
        cur = np.random.choice(self.move_pool, 1, p=self.hist)[0]
        self.clock -= 1
        if self.clock < 1: # if the clock runs out (number of actions made with current pool)
                           # choose a new temporary move pool and probability
            self.reset_hist()
        return cur


    def reset_hist(self):
        num = np.random.choice(self.pool_sizes, 1, p=self.probs)  # choose a random number of active moves

        # Randomly select that number of moves from the action space, does not allow duplicates
        self.move_pool = np.random.choice(self.action_space, num, False)  # This random choice is UNIFORM and gives the
                                                                          # algorithm its macro uniform distribution
        # Generate random weights for a random histogram
        weights = np.random.choice([1, 2, 3], num)  # This binds the range of probability to a range of positive values
                                                    # given [1,2,3] the range of probabilties is 1/7 - 3/5 or 14.2% - 60%
        self.hist = weights / weights.sum() # normalizing the weights into a set of probabilties
        self.clock = self.duration  # reset the clock

