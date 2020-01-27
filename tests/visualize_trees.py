import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()

resolution = 50  # the number of vertices
radii = 0.05
patches = []
patches.append(Circle((0.5, 0.8), radii))
patches.append(Circle((0.25, 0.5), radii))
patches.append(Circle((0.75, 0.5), radii))

colors = 100*np.random.rand(len(patches))
p = PatchCollection(patches, alpha=0.8)
p.set_array(np.array(colors))
ax.add_collection(p)
fig.colorbar(p, ax=ax)

plt.axis('off')
plt.show()