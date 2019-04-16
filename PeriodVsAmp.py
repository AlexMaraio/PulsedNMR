'''
    This file is responsible for producing the plots that allow us to set a 90% and 95%
    limit on the minimum period required to still obtain a good FID signal.
'''
#! Importing the stuff necessary
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
import scipy.optimize as sciopt

#! The rough fitting function that the data can be approximated to
def FitFunc(t,A,tau):
    return A*(1 - np.exp(-t/tau))

#! Data taken
Periods = np.concatenate((np.linspace(50.0,200.0,16),np.linspace(220.0,280.0,4)))
MaxAmp = [6.91,7.42,7.81,8.19,8.51,8.77,9.02,9.22,9.41,9.54,9.73,9.86,9.98,10.0,10.1,10.1,10.3,10.4,10.5,10.6]

#! Performing a rough fit to the data of a rising exponential decay
InitParams = [12,10]
Fit, Errs = sciopt.curve_fit(FitFunc,Periods,MaxAmp,p0=InitParams)
print(Fit)

#! Plotting code
PeriodLinspace = np.linspace(0,max(Periods),10000)
plt.plot(Periods,MaxAmp,label="Data")
plt.plot(PeriodLinspace,FitFunc(PeriodLinspace, *Fit),label='Rough Fit')
plt.xlim(xmin=0,xmax=max(Periods))
plt.ylim(ymin=0)
plt.xlabel('Period of the pulse train (s)')
plt.ylabel('Max amplitude of FID (V)')
plt.title('Plot of how the amplitude of the FID changes with period')
#plt.axhline(y=0.90*max(MaxAmp),color='red')
#*plt.axhspan(ymin=0.90*max(MaxAmp)-0.005,ymax=0.90*max(MaxAmp)+0.005,xmin=0,xmax=140/max(Periods),color='red',label='90% condfidance')
#*plt.axhline(y=0.95*max(MaxAmp),color='blue')
plt.legend()
plt.show()