import multiprocess
import explore_Viewer


def launch_viewer(q, name="Explore", x=5, y=20, size=21, window_height=300):
    view = explore_Viewer.exploreViewer(q, name, x, y, size, window_height)
    view.start()

