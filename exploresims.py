import numpy as np
import time
from multiprocess import Process, Queue, Semaphore, Lock
from explore import ExploreGame, launch_viewer
from randomchooser import Chooser

def simulate_explore(semaphore, out_lock, q_out, samples, games, turns, chooser, size=21, window_height=300, framerate=121, acceleration=1, animate=True, position_queue=None ):
    semaphore.acquire()
    env = ExploreGame(size)  # create our game environment
    view_name = "Explore" + str(chooser.weights) + " : " + str(chooser.duration)
    if animate:
        if position_queue is None:
            raise Exception("Can't animate sim without position queue!")
        coords = position_queue.get()
        view_x = coords[0]
        view_y = coords[1]
        q_animate = Queue()
        myviewer = Process(target=launch_viewer, args=[q_animate, view_name, view_x, view_y, size, window_height])
        q_animate.put((env.generate_frame(), env.get_trace()))
        myviewer.start()
        time.sleep(3)  # let the window init
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
            wins += env.score
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
        position_queue.put(coords)
    semaphore.release()
    result_trace /= samples
    result_wins = np.array(result_wins)
    result_trace_stdevs = np.array(result_trace_stdevs)
    sqrsamp = samples**.5
    win_data = (result_wins.mean(), result_wins.std(), result_wins.std()/sqrsamp)
    stdev_data = (result_trace_stdevs.mean(), result_trace_stdevs.std(), result_trace_stdevs.std()/sqrsamp)
    outputs = (chooser.weights, chooser.duration, win_data, stdev_data, result_trace)
    print(view_name, "Experiment ended")
    out_lock.acquire()
    q_out.put(outputs)
    out_lock.release()


def sim_1(choosers, samples=100, games=100, turns=80, num_process=4, num_column=2):
    max_processes = num_process
    sem = Semaphore(max_processes)
    out_lock = Lock()
    output_queue = Queue()
    pos_queue = Queue()

    # make positions available
    num_cols = num_column
    for i in range(max_processes):
        x = 30 + (i % num_cols) * (610 + 30)
        y = 40 + (i // num_cols) * (300 + 40)
        pos_queue.put((x, y))
    sims = []
    for ch in choosers:
        curr_args = [sem, out_lock, output_queue, samples, games, turns, ch, 21,
                     300, 30, 1.05, True, pos_queue]
        p = Process(target=simulate_explore, args=curr_args)
        sims.append(p)

    for sim in sims:
        sim.start()

    results = []
    tgt = len(choosers)
    while len(results) < tgt:
        out_lock.acquire()
        if not output_queue.empty():
            results.append(output_queue.get())
        out_lock.release()
        time.sleep(2)  # only check once ever 2 seconds

    return results

def demo_1(samples=1, games=10, turns=80, num_process=1, num_column=2):
    max_processes = num_process
    sem = Semaphore(max_processes)
    out_lock = Lock()
    output_queue = Queue()
    pos_queue = Queue()

    choosers = [Chooser(5)]

    # make positions available
    num_cols = num_column
    for i in range(max_processes):
        x = 30 + (i % num_cols) * (610 + 30)
        y = 40 + (i // num_cols) * (300 + 40)
        pos_queue.put((x, y))
    sims = []
    for ch in choosers:
        curr_args = [sem, out_lock, output_queue, samples, games, turns, ch, 21,
                     300, 15, 1.0, True, pos_queue]
        p = Process(target=simulate_explore, args=curr_args)
        sims.append(p)

    for sim in sims:
        sim.start()

    results = []
    tgt = len(choosers)
    while len(results) < tgt:
        out_lock.acquire()
        if not output_queue.empty():
            results.append(output_queue.get())
        out_lock.release()
        time.sleep(2)  # only check once ever 2 seconds

    return None
