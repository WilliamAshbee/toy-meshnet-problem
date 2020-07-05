from matplotlib.patches import Circle
import matplotlib.pyplot as plt
e = Circle( xy=(0,0), radius=1 )
ax = plt.gca()  # ax = subplot( 1,1,1 )
ax.add_artist(e)
e.set_clip_box(ax.bbox)
e.set_edgecolor( "black" )
e.set_facecolor( "none" )  # "none" not None
e.set_alpha( 1 )
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight')
