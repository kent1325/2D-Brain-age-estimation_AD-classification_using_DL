import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

colors = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]

mpl.style.use("tableau-colorblind10")
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams["lines.markersize"] = 8
mpl.rcParams["figure.figsize"] = (10, 5)
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "lightgray"
mpl.rcParams["axes.prop_cycle"] = colors
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.titlesize"] = 25
mpl.rcParams["figure.dpi"] = 100