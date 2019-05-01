import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
from lmfit import Model, Parameters
import pandas as pd

#! Matplotlib commands to lock cursor to data and get position of clicks
class Cursor(object):
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        #self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        #self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()
    
class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    """
    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        #self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        #self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        #print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()
        
    def mouse_click(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        ParamsPltX.append(x)
        ParamsPltY.append(y)

ParamsPltX = []
ParamsPltY = []

def onclick_main(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
    ParamsPltX.append(event.xdata)
    ParamsPltY.append(event.ydata)
    bbox = ax1.get_window_extent()
    width, height = bbox.width, bbox.height
    plt.axvline(x=event.xdata)
    plt.axhline(y=event.ydata,xmin=(event.x/width)-0.2,xmax=(event.x/width))
    plt.draw()

def onclick_individual(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
    Xdata.append(event.xdata)
    Ydata.append(event.ydata)


#* Imports the light mineral oil data from a .CSV file
DataDF =  pd.read_csv("./Data/T2/HMO_AfterEaster.csv",names = ['Time','Voltage'],usecols = [3,4])
#DataDF['Time'] = DataDF['Time'] -0.00010


fig1, ax1 = plt.subplots()
ax1.plot(DataDF['Time'],DataDF['Voltage'])
cursor = SnaptoCursor(ax1, DataDF['Time'], DataDF['Voltage'])
plt.connect('motion_notify_event', cursor.mouse_move)

fig1.canvas.mpl_connect('button_press_event', cursor.mouse_click)
plt.axis([min(DataDF['Time']), max(DataDF['Time']), min(DataDF['Voltage'])-0.2, max(DataDF['Voltage'])+0.2])
plt.show(block=True)
#plt.pause(30)
plt.close()