import numpy as np

class Chooser():
    def __init__(self, space, probs=None, duration=30):
        self.space = space
        self.duration = -1
        self.weights = probs
        if probs is not None and space != len(probs):
            raise Exception("Action space and probability must be the same dimension!")
        if probs is not None:
            self.action_space = np.arange(space)
            self.pool_sizes = np.arange(space)+1
            # The following vars only matter for a variable chooser
            # These are soft constant params

            # This is a probability distribution that will be used to decide how many
            # actions will be available to choose from at a given time. Ranging from 1 - space.
            self.probs = np.array(probs)

            self.probs = self.probs / self.probs.sum()
            self.duration = duration

            # These are volatile/dynamic params that may change randomly over time
            self.clock = duration  # acts as a timer for when to randomize the move pool and hist
            self.move_pool = np.arange(space)
            self.hist = np.ones(space)*10
            self.reset_hist()
        else:
            self.probs = None

    def get_random_action(self):  # Calls the appropriate action "getter" function based on mode set on init
        if self.probs is None:
            return self.get_uniformly_random_action()  # the standard random choice function
        else:
            return self.get_variably_random_action()  # the experimental choice function

    def get_uniformly_random_action(self):  # returns a single random action choice from the initialized action space
        return np.random.choice(self.space)

    def get_variably_random_action(self):
        # returns a random action choice from a temporary action space that may be a subset of size 1 - n of the true
        # action space. It also uses a temporary probability distribution to bias choices for each chosen subset.
        # Each temporary action pool and histogram is only used for a fixed number of calls before being, randomly
        # regenerated.

        # pull a move from the current move pool with the current probability distribution
        cur = np.random.choice(self.move_pool, 1, p=self.hist)[0]
        self.clock -= 1
        if self.clock < 1:
            # if the clock runs out (number of actions made with current pool)
            # choose a new temporary move pool and probability
            self.reset_hist()
        return cur

    def reset_hist(self):  # Generates anew temporary move pool and probability distribution pair
        num = np.random.choice(self.pool_sizes, 1, p=self.probs)  # choose a random number of active moves

        # Randomly select that number of moves from the action space, does not allow duplicates
        # This random choice is ALWAYS UNIFORM and gives the algorithm its macro uniform distribution
        self.move_pool = np.random.choice(self.action_space, num, False)
        # Generate random weights for a random histogram
        weights = np.random.choice([1, 2, 3], num)
        # This binds the range of probability to a range of positive values given weight choices [1,2,3] and num
        # the range of probabilities is 1/7 - 3/5 or 14.2% - 60% - i.e. the combination cases of [1,3,3] and [3,1,1]
        # when num is 3 the generalized min is 1 /(1 + 3*(num-1)) and the generalized max is 3/(3 + 1*(num-1))

        self.hist = weights / weights.sum()  # normalizing the weights into a set of probabilities
        self.clock = self.duration  # reset the clock

