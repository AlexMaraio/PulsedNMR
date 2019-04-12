import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import scipy.stats as scistats
from lmfit import Model, Parameters

def T1Func(time,M0,T1):
    return M0 * ( 1 - 2*np.exp(-time/T1))

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


#! Heavy mineral oil T1 decay time
TimesHMO = np.concatenate((np.linspace(0.0010,0.0220,22), np.linspace(0.0240,0.0600,37)))
AmplitudeHMO = np.array( [ -6.16,-5.74,-5.37,-5.02,-4.68,-4.32,-3.98,-3.65,-3.37,-3.06,-2.77,-2.50,-2.26,-1.99,-1.77,-1.56,-1.36,-1.10,-0.881,-0.686,-0.492,-0.310, \
    0.152,0.330,0.498,0.660,0.808,0.951,1.10,1.23,1.37,1.50,1.63,1.74,1.84,1.96,2.07,2.15,2.26,2.35,2.44,2.53,2.63,2.71,2.79,2.87,2.96,3.03,3.11,3.19,3.27,3.32,3.41,3.47,3.54,3.61,3.65,3.72,3.78 ])
AmplitudeErrorHMO = np.array([0.02]*59)

T1HMOModel = Model(T1Func)
T1HMOParams = Parameters()

T1HMOParams.add('M0',value=7,min=0,vary=True)
T1HMOParams.add('T1',value=3e-3,min=0,vary=True)

T1HMOFit = T1HMOModel.fit(AmplitudeHMO,params=T1HMOParams,time=TimesHMO,weights=1./AmplitudeErrorHMO)
print(T1Fit.fit_report())
T1HMOFit.plot(show_init=False,yerr=AmplitudeErrorHMO,xlabel='Time (s)',ylabel='Net Magnetisation')
plt.show(block=False)


