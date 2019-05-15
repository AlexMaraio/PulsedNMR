'''
    This Python file is all about the T1 decay times, so this is the data taken for
    the first three samples, and from this we can fit a rising exponential to the data,
    and then extract T1, which is the spin-lattice relaxation time.
'''
import matplotlib.pyplot as plt
from matplotlib import container
import numpy as np
import seaborn as sns
sns.set(font_scale=1.35)
import scipy.stats as scistats
from lmfit import Model, Parameters

#? The function that we are trying to fit our data to, as prediced by the theory
def T1Func(time,M0,T1):
    return M0 * ( 1 - 2*np.exp(-time/T1))

#! Light minearal oil data
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
T1Fit.plot(show_init=False,yerr=AmplitudeError,xlabel='Time (s)',ylabel='Net Magnetisation',title="Light Minearal Oil T1")
plt.show(block=False)


plt.figure()
plt.plot(Times,T1Func(Times,T1Fit.best_values['M0'],T1Fit.best_values['T1']),label="Fit",linewidth=2)
#plt.plot(Times,Amplitude,'gx',label="Data",linewidth=2)
plt.errorbar(Times,Amplitude,fmt='gx',yerr=0.08,label="Data",linewidth=2,markersize=7.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Net Magnetisation as a Function of Time')

ax = plt.gca()

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

ax.legend(handles, labels)
#plt.legend()
plt.show()



#! Heavy mineral oil T1 decay time
TimesHMO = np.concatenate((np.linspace(0.0010,0.0220,22), np.linspace(0.0240,0.0600,37)))
AmplitudeHMO = np.array( [ -6.16,-5.74,-5.37,-5.02,-4.68,-4.32,-3.98,-3.65,-3.37,-3.06,-2.77,-2.50,-2.26,-1.99,-1.77,-1.56,-1.36,-1.10,-0.881,-0.686,-0.492,-0.310, \
    0.152,0.330,0.498,0.660,0.808,0.951,1.10,1.23,1.37,1.50,1.63,1.74,1.84,1.96,2.07,2.15,2.26,2.35,2.44,2.53,2.63,2.71,2.79,2.87,2.96,3.03,3.11,3.19,3.27,3.32,3.41,3.47,3.54,3.61,3.65,3.72,3.78 ])
AmplitudeErrorHMO = np.array([0.02]*59)

#! Now going from slightly later to see if this reduces any error
TimesHMO = np.concatenate((np.linspace(0.0070,0.0220,16), np.linspace(0.0240,0.0600,37)))
AmplitudeHMO = np.array( [-3.98,-3.65,-3.37,-3.06,-2.77,-2.50,-2.26,-1.99,-1.77,-1.56,-1.36,-1.10,-0.881,-0.686,-0.492,-0.310, \
    0.152,0.330,0.498,0.660,0.808,0.951,1.10,1.23,1.37,1.50,1.63,1.74,1.84,1.96,2.07,2.15,2.26,2.35,2.44,2.53,2.63,2.71,2.79,2.87,2.96,3.03,3.11,3.19,3.27,3.32,3.41,3.47,3.54,3.61,3.65,3.72,3.78 ])
AmplitudeErrorHMO = np.array([0.02]*53)

T1HMOModel = Model(T1Func)
T1HMOParams = Parameters()

T1HMOParams.add('M0',value=7,min=0,vary=True)
T1HMOParams.add('T1',value=3e-3,min=0,vary=True)

T1HMOFit = T1HMOModel.fit(AmplitudeHMO,params=T1HMOParams,time=TimesHMO,weights=1./AmplitudeErrorHMO)
print(T1HMOFit.fit_report())
T1HMOFit.plot(show_init=False,yerr=AmplitudeErrorHMO,xlabel='Time (s)',ylabel='Net Magnetisation',title="Heavy Minearal Oil T1")
plt.show(block=False)


#! Glycerin T1 decay times
TimesGlycerin = np.concatenate((np.linspace(10,32,23), np.linspace(34,58,25))) *1e-3
AmplitudeGlycerin = np.array( [ -3.56,-3.37,-3.16,-2.94,-2.78,-2.59,-2.43,-2.28,-2.12,-1.94,-1.78,-1.65,-1.50,-1.35,-1.18,-1.05,-0.905,-0.776,-0.640,-0.496,-0.370,-0.241,-0.119, \
    0.122,0.227,0.350,0.455,0.567,0.673,0.773,0.882,0.973,1.07,1.16,1.25,1.35,1.45,1.52,1.62,1.68,1.76,1.85,1.93,1.98,2.05,2.12,2.20,2.24] )
AmplitudeErrorGlycerin = np.array([0.02]*48)

T1GlycerinModel = Model(T1Func)
T1GlycerinParams = Parameters()

T1GlycerinParams.add('M0',value=7,min=0,vary=True)
T1GlycerinParams.add('T1',value=3e-3,min=0,vary=True)

T1GlycerinFit = T1GlycerinModel.fit(AmplitudeGlycerin,params=T1GlycerinParams,time=TimesGlycerin,weights=1./AmplitudeErrorGlycerin)
print(T1GlycerinFit.fit_report())
T1GlycerinFit.plot(show_init=False,yerr=AmplitudeErrorGlycerin,xlabel='Time (s)',ylabel='Net Magnetisation',title="Glycerin T1")
plt.show(block=False)

