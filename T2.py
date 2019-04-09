import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import scipy.stats as scistats
from lmfit import Model, Parameters
import pandas as pd

def T2Func(time,A,T2):
    return A * np.exp(-time/T2)

DataDF =  pd.read_csv("./Data/T2/NewLMOTraceT2.csv",names = ['Time','Voltage'],usecols = [3,4])
DataDF['Time'] = DataDF['Time'] -0.00010

plt.plot(DataDF['Time'],DataDF['Voltage'])
plt.title('Two-pulse spin-echo experiments to find T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')



DataTime = np.array([ 0.0,5.0,10.25,15.40,20.60,25.90,31.05,36.20,41.45,46.60,51.85,57.00,62.20,67.40,72.40,77.80,83.10,88.30,93.40,98.60,103.80,109.20,114.30,119.50,124.70,129.85,135.00,140.10,145.50,150.70,156.00 ]) * 1e-3
DataAmp = np.array([11.040,10.080,9.065,8.225,7.443,6.840,6.305,5.760,5.230,4.805,4.380,4.025,3.730,3.490,3.240,3.000,2.820,2.580,2.400,2.225,2.103,1.980,1.860,1.740,1.620,1.560,1.440,1.440,1.325,1.260,1.200])

plt.plot(DataTime,DataAmp,'gx')
plt.show()



T2Model = Model(T2Func)
T2Params = Parameters()

T2Params.add('A',value=7,min=0,vary=True)
T2Params.add('T2',value=3e-3,min=0,vary=True)

T2Fit = T2Model.fit(DataAmp,params=T2Params,time=DataTime,weights=1/0.1)
print(T2Fit.fit_report())
T2Fit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.1)
plt.show()



'''
Times = np.concatenate((np.linspace(0.002,0.0360,35), np.linspace(0.0380,0.062,25)))

Amplitude = [ -7.88,-7.57,-7.26,-6.95,-6.69,-6.38,-6.12,-5.90,-5.59,-5.32,-5.06,-4.80,-4.53,-4.27,-4.05,-3.78,-3.56,-3.34,-3.12,-2.86,-2.68,-2.46,-2.29,-2.07,-1.89,-1.71, \
    -1.54,-1.37,-1.19,-1.01,-0.892,-0.670,-0.512,-0.337,-0.174,0.245,0.394,0.552,0.691,0.834,0.973,1.11,1.24,1.38,1.50,1.61,1.74,1.86,1.97,2.07,2.20,2.31,2.41, \
        2.52,2.62,2.72,2.82,2.93,3.02,3.14 ]

AmplitudeError = np.concatenate(([0.08]*30, [0.04]*11, [0.08]*19))
#! The 0.08 comes from the error on the cursor measurements, as this is the minimum value that we can increment the cursor by, on the y-scale.
#! The 0.04 comes from the fact that we zoomed in on the oscilloscope to get this data, and so the minimum increment of the cursor would have decreased, 
#! however, the line on the oscilloscope becomes noticibly thicker, and so obtaining the location of the peak becomes harder, and so a greater error.

T1Model = Model(T1Func)
T1Params = Parameters()

T1Params.add('M0',value=7,min=0,vary=True)
T1Params.add('T1',value=3e-3,min=0,vary=True)

T1Fit = T1Model.fit(Amplitude,params=T1Params,time=Times,weights=1./AmplitudeError)
print(T1Fit.fit_report())
T1Fit.plot(show_init=False,yerr=AmplitudeError,xlabel='Time (s)',ylabel='Net Magnetisation')
plt.show()
#plt.show(block=False)


Periods = np.concatenate((np.linspace(50.0,200.0,16),np.linspace(220.0,280.0,4)))
MaxAmp = [6.91,7.42,7.81,8.19,8.51,8.77,9.02,9.22,9.41,9.54,9.73,9.86,9.98,10.0,10.1,10.1,10.3,10.4,10.5,10.6]

plt.plot(Periods,MaxAmp)
plt.show()
'''