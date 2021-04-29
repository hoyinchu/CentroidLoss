import numpy as np
import matplotlib.pyplot as plt

def paint_centroids(points,centroids,radii,point_colors,centroid_colors):
    assert len(centroids) == len(radii)

    circle_plts = []
    if centroid_colors == None:
        centroid_colors = ['black'] * len(centroids)
    if len(points) > 0 and point_colors == None:
        point_colors = ['black'] * len(points)
    
    for i in range(len(centroids)):
        cur_circle_plt = plt.Circle(centroids[i], radii[i], color=centroid_colors[i], fill=False)
        circle_plts.append(cur_circle_plt)

    # circle1 = plt.Circle((0, 0), 0.2, color='r')
    # circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
    # circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    for i in range(len(centroids)):
        ax.add_patch(circle_plts[i])
    
    if len(points) > 0:
        ax.scatter(points[:,0],points[:,1],marker='x',c=point_colors)
    
    # (or if you have an existing figure)
    # fig = plt.gcf()
    # ax = fig.gca()

    plt.plot()