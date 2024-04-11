import numpy as np
import time
import multiprocess
import exploreModel
from randomchooser import Chooser

def simulate_explore(samples, games, turns, chooser, q_out, q_animate=None, framerate=None):
    env = exploreModel.exploreGame() # create our game environment

    if q_animate is not None: # populate viewer with start state
        q_animate.put((env.generate_frame(), env.get_trace()))
    result_wins = []
    result_trace_stdevs = []
    result_trace = np.zeros((env.size, env.size))
    for k in range(samples):
        wins = 0
        for j in range(games):
            for i in range(turns):
               env.update(chooser.get_random_action())
               if q_animate is not None and q_animate.empty(): # if we are rendering send viewer frame
                   q_animate.put((env.generate_frame(), env.get_trace()))
               if framerate is not None:
                   time.sleep(1/framerate)
            #if we hit the goal
            if env.won:
                wins += 1
            env.soft_reset()
        result_wins.append(wins)
        trace = env.get_trace()
        result_trace_stdevs.append(trace.std())
        result_trace += trace
        env.hard_reset()

    result_trace /= samples
    output = (result_wins, result_trace_stdevs, result_trace)
    q_out.put(output)


