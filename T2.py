import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
import scipy.stats as scistats
from lmfit import Model, Parameters
import pandas as pd
import scipy.signal as scisig
from termcolor import cprint #* coloured terminal printing -- might be a pain on windows *_*


TitleFont = {'size':'24', 'color':'black', 'weight':'bold'} 
AxTitleFont = {'size':'16'}

#* Simple exponential decay curve for fitting
def T2Func(time,A,T2):
    return A * np.exp(-time/T2)

#* Linear sum of two exponential functions to fit to.
def TwoExponential(time,A1,T2_1, A2, T2_2):
    return A1 * np.exp(-time/T2_1) + A2 * np.exp(-time/T2_2)

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


#* Imports the light mineral oil data from a .CSV file
DataDF =  pd.read_csv("./Data/T2/LMOThurs11Data.csv",names = ['Time','Voltage'],usecols = [3,4])

#* Plots the light mineral oul data, so that way the location of the peaks can be found.
fig1, ax1 = plt.subplots()
ax1.plot(DataDF['Time'],DataDF['Voltage'])
cursor = SnaptoCursor(ax1, DataDF['Time'], DataDF['Voltage'])
plt.connect('motion_notify_event', cursor.mouse_move)

fig1.canvas.mpl_connect('button_press_event', cursor.mouse_click)
plt.axis([min(DataDF['Time']), max(DataDF['Time']), min(DataDF['Voltage'])-0.2, max(DataDF['Voltage'])+0.2])
plt.show(block=True)
#plt.pause(30)
plt.close()


plt.plot(DataDF['Time'],DataDF['Voltage'])
plt.title('Two-pulse spin-echo experiments to find T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')


DataTime = ParamsPltX 
DataAmp = ParamsPltY

#! Start of actual data fitting that is useful!

#* Last 9 data points for light minearal oil
DataTime = np.array( [ 98.40,114.70,131.20,147.50,163.60,180.2,196.50,213.10,229.40]) *1e-3
DataAmp = np.array( [ 1.15,0.94,0.81,0.65,0.56,0.50,0.41,0.37,0.31] )

#* First 4 data points for light minearal oil
DataTime2 = np.array( [ 0.10,16.30,32.80,49.0])*1e-3
DataAmp2 = np.array( [ 5.40,3.87,2.84,2.15] )

plt.plot(DataTime,DataAmp,'gx')
plt.plot(DataTime2,DataAmp2,'yx')
plt.show(block=False)

# Fitting a single exponential function to the last nine data points for light minearal oil
T2Model = Model(T2Func)
T2Params = Parameters()
T2Params.add('A',value=4.60,min=1e-10,vary=True)
T2Params.add('T2',value=0.073,min=1e-10,vary=True)
T2Fit = T2Model.fit(DataAmp,params=T2Params,time=DataTime,weights=1/0.02)
cprint('Light mineral oil, single exponential, last nine',color='cyan',on_color='on_magenta',attrs=['bold'])
print(T2Fit.fit_report())
T2Fit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Light mineral oil - Single exponential - last nine')
plt.show(block=False)

# Now fitting a single exponential function, but now to the first four data points for LMO for comparison
T2Model2 = Model(T2Func)
T2Params2 = Parameters()
T2Params2.add('A',value=4.60,min=1e-10,vary=True)
T2Params2.add('T2',value=0.073,min=1e-10,vary=True)
T2Fit2 = T2Model2.fit(DataAmp2,params=T2Params2,time=DataTime2,weights=1/0.02)
cprint('Light mineral oil, single exponential, first four',color='cyan',on_color='on_magenta',attrs=['bold'])
print(T2Fit2.fit_report())
T2Fit2.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Light mineral oil - Single exponential - first four')
plt.show(block=False)


#* Plotting the comparison of the different fits
plt.figure()
plt.plot(DataDF['Time'],DataDF['Voltage'])
plt.plot(DataDF['Time'],T2Func(DataDF['Time'],T2Fit.best_values['A'],T2Fit.best_values['T2']))
plt.plot(DataDF['Time'],T2Func(DataDF['Time'],T2Fit2.best_values['A'],T2Fit2.best_values['T2']))
plt.title('Two-pulse spin-echo experiments to find T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show(block=False)


#? Now fitting the addition of the two exponential functions for comparison

#* The full data array of every T2 peak
DataTimeFull = np.array( [ 0.10,16.30,32.80,49.0, 65.50, 81.80,   98.40,114.70,131.20,147.50,163.60,180.2,196.50,213.10,229.40]) *1e-3
DataAmpFull = np.array( [ 5.40,3.87,2.84,2.15, 1.72, 1.37, 1.15,0.94,0.81,0.65,0.56,0.50,0.41,0.37,0.31] )

#* Single exponential fit, all data
T2ModelSingle = Model(T2Func) 
T2ParamsSingle = Parameters()

T2ParamsSingle.add('A',value=4,min=1e-10,vary=True)
T2ParamsSingle.add('T2',value=0.01,min=1e-10,vary=True)

T2FitSingle = T2ModelSingle.fit(DataAmpFull,params=T2ParamsSingle,time=DataTimeFull,weights=1/0.02)
cprint('Light mineral oil, one exponentials, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(T2FitSingle.fit_report())
T2FitSingle.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Light mineral oil - Single exponential - all data')
plt.show(block=False)

#* Double exponential fit, all data
T2ModelSum = Model(TwoExponential) 
T2ParamsSum = Parameters()

T2ParamsSum.add('A1',value=4,min=1e-10,vary=True)
T2ParamsSum.add('T2_1',value=0.01,min=1e-10,vary=True)
T2ParamsSum.add('A2',value=5,min=1e-10,vary=True)
T2ParamsSum.add('T2_2',value=0.1,min=1e-10,vary=True)

T2FitSum = T2ModelSum.fit(DataAmpFull,params=T2ParamsSum,time=DataTimeFull,weights=1/0.02)
cprint('Light mineral oil, two exponentials, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print('\n Summation of two peaks fit, light minearal oil \n' + T2FitSum.fit_report())
T2FitSum.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Light mineral oil - Two exponentials - all data')
plt.show(block=False)


#* Plotting the comparisons between all of the fits for the light minearal oil
plt.figure()
plt.plot(DataDF['Time'],DataDF['Voltage'],label="Data")
plt.plot(DataTimeFull,DataAmpFull,'gx')
plt.plot(DataDF['Time'],T2Func(DataDF['Time'],T2Fit.best_values['A'],T2Fit.best_values['T2']),label="Last 9 Fit",color='green')
plt.plot(DataDF['Time'],T2Func(DataDF['Time'],T2Fit2.best_values['A'],T2Fit2.best_values['T2']),label="First 4 Fit",color='#7A0E71')
plt.plot(DataDF['Time'],T2Func(DataDF['Time'],T2FitSingle.best_values['A'],T2FitSingle.best_values['T2']),label="1 Exp Fit")
plt.plot(DataDF['Time'],TwoExponential(DataDF['Time'],T2FitSum.best_values['A1'],T2FitSum.best_values['T2_1'],T2FitSum.best_values['A2'],T2FitSum.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of different fits for light minearal oil')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=True)


#? Heavy Minearal Oil T2 Calculations
HMODF =  pd.read_csv("./Data/T2/HMOTraceT2Fri.csv",names = ['Time','Voltage'],usecols = [3,4])

HMOtime = np.array([ 0.10,6.80,13.60,20.40,27.20,33.98,40.80,47.62,54.42,61.20,68.00,74.70,81.60,88.40,95.20 ]) * 1e-3
HMOamp = np.array( [ 6.56,4.60,3.36,2.56,2.00,1.64,1.36,1.16,0.96,0.80,0.72,0.60,0.56,0.48,0.42 ])

#* Plotting the Heavy mineral oil data
plt.plot(HMODF['Time'],HMODF['Voltage'])
plt.plot(HMOtime,HMOamp,'rx')
plt.title('Heavy Minearal Oil T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show(block=False)

#* Fitting the heavy minearal oil data

#? Single exponential
HMOT2Model = Model(T2Func)
HMOT2Params = Parameters()
HMOT2Params.add('A',value=4.60,min=1e-10,vary=True)
HMOT2Params.add('T2',value=0.073,min=1e-10,vary=True)
HMOT2Fit = HMOT2Model.fit(HMOamp,params=HMOT2Params,time=HMOtime,weights=1/0.02)
cprint('Heavy mineral oil, single exponential, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(HMOT2Fit.fit_report())
HMOT2Fit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Heavy Minearal Oil T2 - Single Exponential')
plt.show(block=False)

#? Two exponentials
HMOT2Model2 = Model(TwoExponential) 
HMOT2Params2 = Parameters()
HMOT2Params2.add('A1',value=4,min=1e-10,vary=True)
HMOT2Params2.add('T2_1',value=0.01,min=1e-10,vary=True)
HMOT2Params2.add('A2',value=5,min=1e-10,vary=True)
HMOT2Params2.add('T2_2',value=0.1,min=1e-10,vary=True)
HMOT2Fit2 = HMOT2Model2.fit(HMOamp,params=HMOT2Params2,time=HMOtime,weights=1/0.02)
cprint('Heavy mineral oil, two exponentials, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(HMOT2Fit2.fit_report())
HMOT2Fit2.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Heavy Minearal Oil T2 - Two Exponentials')
plt.show(block=False)


#! Plotting the single and double exponential functions to compare
plt.figure()
plt.plot(HMODF['Time'],HMODF['Voltage'],label="Data")
plt.plot(HMOtime,HMOamp,'gx')
plt.plot(HMODF['Time'],T2Func(HMODF['Time'],HMOT2Fit.best_values['A'],HMOT2Fit.best_values['T2']),label="1 Exp Fit",color='green')
plt.plot(HMODF['Time'],TwoExponential(HMODF['Time'],HMOT2Fit2.best_values['A1'],HMOT2Fit2.best_values['T2_1'],HMOT2Fit2.best_values['A2'],HMOT2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of single vs two exponential fit of HMO for T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=False)


#* Now doing the same comparison for the first four data points and the last nine data points, just like for the light minearal oil


#? First four
HMOtimeFirstFour = np.array([ 0.10,6.80,13.60,20.40 ]) * 1e-3
HMOampFirstFour = np.array( [ 6.56,4.60,3.36,2.56 ])
HMOT2FirstFourModel = Model(T2Func)
HMOT2FirstFourParams = Parameters()
HMOT2FirstFourParams.add('A',value=4.60,min=1e-10,vary=True)
HMOT2FirstFourParams.add('T2',value=0.073,min=1e-10,vary=True)

HMOT2FirstFourFit = HMOT2FirstFourModel.fit(HMOampFirstFour,params=HMOT2FirstFourParams,time=HMOtimeFirstFour,weights=1/0.02)
cprint('Heavy mineral oil, single exponential, first four data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(HMOT2FirstFourFit.fit_report())
HMOT2FirstFourFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Heavy Minearal Oil T2 - First Four Single Exp')
plt.show(block=False)

#? Last nine
HMOtimeLastNine = np.array([ 40.80,47.62,54.42,61.20,68.00,74.70,81.60,88.40,95.20 ]) * 1e-3
HMOampLastNine = np.array( [ 1.36,1.16,0.96,0.80,0.72,0.60,0.56,0.48,0.42 ])

HMOT2LastNineModel = Model(T2Func)
HMOT2LastNineParams = Parameters()
HMOT2LastNineParams.add('A',value=4.60,min=1e-10,vary=True)
HMOT2LastNineParams.add('T2',value=0.073,min=1e-10,vary=True)
HMOT2LastNineFit = HMOT2FirstFourModel.fit(HMOampLastNine,params=HMOT2LastNineParams,time=HMOtimeLastNine,weights=1/0.02)
cprint('Heavy mineral oil, single exponential, last nine data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(HMOT2LastNineFit.fit_report())
HMOT2LastNineFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Heavy Minearal Oil T2 - Last Nine Single Exp')
plt.show(block=False)

#! Comparison plot between the three different fits for heavy minearal oil
plt.figure()
plt.plot(HMODF['Time'],HMODF['Voltage'],label="Data")
plt.plot(HMOtime,HMOamp,'gx')
plt.plot(HMODF['Time'],T2Func(HMODF['Time'],HMOT2FirstFourFit.best_values['A'],HMOT2FirstFourFit.best_values['T2']),label="First Four",color='#7A0E71')
plt.plot(HMODF['Time'],T2Func(HMODF['Time'],HMOT2LastNineFit.best_values['A'],HMOT2LastNineFit.best_values['T2']),label="Last Nine",color='green')
plt.plot(HMODF['Time'],T2Func(HMODF['Time'],HMOT2Fit.best_values['A'],HMOT2Fit.best_values['T2']),label="1 Exp Fit",color='orange')
plt.plot(HMODF['Time'],TwoExponential(HMODF['Time'],HMOT2Fit2.best_values['A1'],HMOT2Fit2.best_values['T2_1'],HMOT2Fit2.best_values['A2'],HMOT2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of different fits for heavy minearal oil')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=True)


#? Glycerin sample now 

#* Importing the glycerin T2 data
GlycerinDF =  pd.read_csv("./Data/T2/GlycerinT2Fri.csv",names = ['Time','Voltage'],usecols = [3,4])

GlycerinTime = np.array([ 0.10,6.00,12.00,18.00,24.00,29.95,36.00,42.00,48.00,54.00,60.00,65.95,72.00,77.84,84.00,89.85,96.00 ]) * 1e-3
GlycerinAmp = np.array( [ 5.72,4.96,4.24,3.56,3.00,2.52,2.16,1.88,1.64,1.40,1.20,1.04,0.92,0.80,0.72,0.60,0.56 ])

#* Plotting the Heavy mineral oil data
plt.figure()
plt.plot(GlycerinDF['Time'],GlycerinDF['Voltage'])
plt.plot(GlycerinTime,GlycerinAmp,'rx')
plt.title('Glycerin T2 Data')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show(block=False)


#* Fitting the heavy minearal oil data

#? Single exponential
GlycerinT2Model = Model(T2Func)
GlycerinT2Params = Parameters()

GlycerinT2Params.add('A',value=4.60,min=1e-10,vary=True)
GlycerinT2Params.add('T2',value=0.073,min=1e-10,vary=True)

GlycerinT2Fit = GlycerinT2Model.fit(GlycerinAmp,params=GlycerinT2Params,time=GlycerinTime,weights=1/0.02)
cprint('Glycerin, single exponential, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(GlycerinT2Fit.fit_report())
GlycerinT2Fit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Glycerin T2 - Single Exponential')
plt.show(block=False)

#? Two exponentials
GlycerinT2Model2 = Model(TwoExponential) 
GlycerinT2Params2 = Parameters()
GlycerinT2Params2.add('A1',value=4,min=1e-10,vary=True)
GlycerinT2Params2.add('T2_1',value=0.01,min=1e-10,vary=True)
GlycerinT2Params2.add('A2',value=5,min=1e-10,vary=True)
GlycerinT2Params2.add('T2_2',value=0.1,min=1e-10,vary=True)

GlycerinT2Fit2 = GlycerinT2Model2.fit(GlycerinAmp,params=GlycerinT2Params2,time=GlycerinTime,weights=1/0.02)
cprint('Glycerin, two exponentials, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(GlycerinT2Fit2.fit_report())
GlycerinT2Fit2.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Glycerin T2 - Two Exponentials')
plt.show(block=False)


#! Plotting the single and double exponential functions to compare
plt.figure()
plt.plot(GlycerinDF['Time'],GlycerinDF['Voltage'],label="Data")
plt.plot(GlycerinTime,GlycerinAmp,'gx')
plt.plot(GlycerinDF['Time'],T2Func(GlycerinDF['Time'],GlycerinT2Fit.best_values['A'],GlycerinT2Fit.best_values['T2']),label="1 Exp Fit",color='green')
plt.plot(GlycerinDF['Time'],TwoExponential(GlycerinDF['Time'],GlycerinT2Fit2.best_values['A1'],GlycerinT2Fit2.best_values['T2_1'],GlycerinT2Fit2.best_values['A2'],GlycerinT2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of single vs two exponential fit of Glycerin for T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=False)


#* Now doing the same comparison for the first four data points and the last nine data points, just like for the light minearal oil
#? First four

GlycerinTimeFirstFour = np.array([ 0.10,6.00,12.00,18.00 ]) * 1e-3
GlycerinAmpFirstFour = np.array( [ 5.72,4.96,4.24,3.56 ])

GlycerinT2FirstFourModel = Model(T2Func)
GlycerinT2FirstFourParams = Parameters()
GlycerinT2FirstFourParams.add('A',value=4.60,min=1e-10,vary=True)
GlycerinT2FirstFourParams.add('T2',value=0.073,min=1e-10,vary=True)
GlycerinT2FirstFourFit = GlycerinT2FirstFourModel.fit(GlycerinAmpFirstFour,params=GlycerinT2FirstFourParams,time=GlycerinTimeFirstFour,weights=1/0.02)
cprint('Glycerin, single exponential, first four data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(GlycerinT2FirstFourFit.fit_report())
GlycerinT2FirstFourFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Glycerin T2 - First Four Single Exp')
plt.show(block=False)

#? Last nine

GlycerinTimeLastNine = np.array([ 48.00,54.00,60.00,65.95,72.00,77.84,84.00,89.85,96.00 ]) * 1e-3
GlycerinAmpLastNine = np.array( [ 1.64,1.40,1.20,1.04,0.92,0.80,0.72,0.60,0.56 ])

GlycerinT2LastNineModel = Model(T2Func)
GlycerinT2LastNineParams = Parameters()
GlycerinT2LastNineParams.add('A',value=4.60,min=1e-10,vary=True)
GlycerinT2LastNineParams.add('T2',value=0.073,min=1e-10,vary=True)
GlycerinT2LastNineFit = GlycerinT2FirstFourModel.fit(GlycerinAmpLastNine,params=GlycerinT2LastNineParams,time=GlycerinTimeLastNine,weights=1/0.02)
cprint('Glycerin, single exponential, last nine data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(GlycerinT2LastNineFit.fit_report())
GlycerinT2LastNineFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='Glycerin T2 - Last Nine Single Exp')
plt.show(block=False)


#! Comparison plot between the three different fits for heavy minearal oil
plt.figure()
plt.plot(GlycerinDF['Time'],GlycerinDF['Voltage'],label="Data")
plt.plot(GlycerinTime,GlycerinAmp,'gx')
plt.plot(GlycerinDF['Time'],T2Func(GlycerinDF['Time'],GlycerinT2FirstFourFit.best_values['A'],GlycerinT2FirstFourFit.best_values['T2']),label="First Four",color='#7A0E71')
plt.plot(GlycerinDF['Time'],T2Func(GlycerinDF['Time'],GlycerinT2LastNineFit.best_values['A'],GlycerinT2LastNineFit.best_values['T2']),label="Last Nine",color='green')
plt.plot(GlycerinDF['Time'],T2Func(GlycerinDF['Time'],GlycerinT2Fit.best_values['A'],GlycerinT2Fit.best_values['T2']),label="1 Exp Fit",color='orange')
plt.plot(GlycerinDF['Time'],TwoExponential(GlycerinDF['Time'],GlycerinT2Fit2.best_values['A1'],GlycerinT2Fit2.best_values['T2_1'],GlycerinT2Fit2.best_values['A2'],GlycerinT2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of different fits for glycerin')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=False)
