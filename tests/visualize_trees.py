import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def caltulate_grid_granularity(tree):
    pass


def label(xy, text):
    y = xy[1]
    # y = xy[1] - 0.01  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


def link_nodes(a, b):
    # add a line
    line2 = mlines.Line2D(a, b, lw=2., alpha=0.9)
    colors = np.linspace(0, 1, 2)
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.add_line(line2)
    return


def add_circle(grid_num):
    circle = mpatches.Circle(grid[grid_num], 0.05, ec="none")
    patches.append(circle)
    label(grid[grid_num], "Circle")


def get_grid(pos_y, pos_x):

    num = pos_x + pos_y * dim_x
    return num


def make_grid(dim_x, dim_y):
    grid = np.mgrid[0.1:0.9:dim_x * 1j, 0.9:0.1:-dim_y * 1j].T
    grid = grid.reshape(dim_y * dim_x, 2)
    return grid


def doit():
    # link_nodes(grid[0], grid[1])
    # link_nodes(grid[2], grid[3])
    # link_nodes(grid[3], grid[4])

    label(grid[0], "0")
    label(grid[1], "1")
    label(grid[2], "2")
    label(grid[3], "3")
    label(grid[4], "4")
    label(grid[5], "5")
    label(grid[6], "6")
    label(grid[7], "7")
    label(grid[8], "8")
    label(grid[9], "9")
    label(grid[10], "10")
    label(grid[11], "11")
    label(grid[12], "12")
    label(grid[13], "13")
    label(grid[14], "14")

    # circle2 = mpatches.Circle(grid[1], 0.05, ec="none")
    # patches.append(circle2)
    # label(grid[1], "Circle2")

    # add_circle(0, 1)
    # numer = get_grid(1, 2)
    # print(numer)
    # add_circle(numer)

    # add a fancy box
    grid_num = get_grid(0,2)
    box_width = 0.1
    box_height = 0.05
    fancybox = mpatches.FancyBboxPatch(
        grid[grid_num] - [box_width / 2, box_height / 2], box_width, box_height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02))
    patches.append(fancybox)
    # label(grid[grid_num], "Ifte")
    #
    # # add a line
    # x, y = np.array([[-0.06, 0.0, 0.1], [0.05, -0.05, 0.05]])
    # line = mlines.Line2D(x + grid[8, 0], y + grid[8, 1], lw=2., alpha=0.9)
    # label(grid[8], "grid 8")
    #
    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    # ax.add_line(line)


fig, ax = plt.subplots()
# create 3x3 grid to plot the artists
dim_x = 5
dim_y = 3
grid = make_grid(dim_x, dim_y)

patches = []

doit()

# plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.show()
