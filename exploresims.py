import numpy as np
import time
from multiprocess import Process, Queue, Semaphore, Lock
from explore import ExploreGame, launch_viewer

def simulate_explore(semaphore, out_lock, q_out, samples, games, turns, chooser, framerate=121, acceleration=1, animate=True):
    semaphore.acquire()
    env = ExploreGame()  # create our game environment
    if animate:
        q_animate = Queue()
        myviewer = Process(target=launch_viewer, args=[q_animate])
        q_animate.put((env.generate_frame(), env.get_trace()))
        myviewer.start()
        time.sleep(1)  # let the window init
    result_wins = []
    result_trace_stdevs = []
    result_trace = np.zeros((env.size, env.size))
    for k in range(samples):
        wins = 0
        for j in range(games):
            for i in range(turns):
                env.update(chooser.get_random_action())
                if animate and q_animate.empty():
                    q_animate.put((env.generate_frame(), env.get_trace()))
                    if framerate < 120:
                        time.sleep(1/framerate)
            wins += env.won
            env.soft_reset()  # new game env each game
            if framerate < 120:
                framerate *= acceleration
        result_wins.append(wins)
        temp = env.get_trace()
        result_trace += temp
        result_trace_stdevs.append(temp.std())
        env.hard_reset()  # new game env and clear trace
    if animate:
        q_animate.put((env.generate_frame(), result_trace))
        time.sleep(5)
        q_animate.put("kill")
        myviewer.join()
    semaphore.release()
    result_trace /= samples
    result_wins = np.array(result_wins)
    result_trace_stdevs = np.array(result_trace_stdevs)
    sqrsamp = samples**.5
    win_data = (result_wins.mean(), result_wins.std(), result_wins.std()/sqrsamp)
    stdev_data = (result_trace_stdevs.mean(), result_trace_stdevs.std(), result_trace_stdevs.std()/sqrsamp)
    outputs = (win_data, stdev_data, result_trace)
    out_lock.acquire()
    q_out.put(outputs)
    out_lock.release()
    return outputs


