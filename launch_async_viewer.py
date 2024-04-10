import explore_Viewer

def launch_viewer(q):
    emap = q.get()
    hmap = q.get()
    view = explore_Viewer.exploreViewer(q, emap, hmap)
    view.start()
