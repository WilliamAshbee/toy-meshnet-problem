from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
e = Circle( xy=(.2,.2), radius=.1 )
ax = plt.gca()  # ax = subplot( 1,1,1 )
ax.add_artist(e)
#print(ax.bbox)
bb = ax.bbox
bb._bbox = Bbox(np.array([[0.0, 0.0], [1.0, 1.0]], float))
e.set_clip_box(ax.bbox)
e.set_edgecolor( "black" )
e.set_facecolor( "none" )  # "none" not None
e.set_alpha( 1 )
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight')
