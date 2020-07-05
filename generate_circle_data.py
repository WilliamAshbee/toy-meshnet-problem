import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Get an example image
import matplotlib.cbook as cbook
image_file = cbook.get_sample_data('grace_hopper.png')
img = plt.imread(image_file)

# Make some example data
x = np.random.rand(5)*img.shape[1]
y = np.random.rand(5)*img.shape[0]

# Create a figure. Equal aspect so circles look circular
#fig,ax = plt.subplots(1)
#ax.set_aspect('equal')
fig = plt.figure()
ax = fig.add_subplot(2,1,1)


# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,yy in zip(x,y):
    circ = Circle((xx,yy),50)
    ax.add_patch(circ)

# Show the image
fig.savefig("test.png")
