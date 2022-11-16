import pypianoroll
import matplotlib.pyplot as plt
PATH = "<PATH_TO_THE_SONG>"
multitrack = pypianoroll.read(PATH)
multitrack.binarize()
track = multitrack
#track = multitrack.tracks[2]
track.trim(0, 200 * multitrack.resolution)
track.binarize()
track.plot(grid_axis = 'off',xtick='off',xticklabel=False)
plt.show()
