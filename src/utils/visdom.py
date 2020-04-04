from visdom import Visdom
import numpy as np


class VisdomLinePlotter(object):
    """
    Plots to Visdom

    Borrowed from: https://github.com/noagarcia/visdom-tutorial
    """
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, plot_name, split_name, title, xlabel, ylabel, x, y):
        if plot_name not in self.plots:
            self.plots[plot_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[plot_name], name=split_name, update='append')