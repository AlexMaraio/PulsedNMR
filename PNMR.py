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

T1Model = Model(T1Func)
T1Params = Parameters()

T1Params.add('M0',value=7,min=0,vary=True)
T1Params.add('T1',value=3e-3,min=0,vary=True)

T1Fit = T1Model.fit(Amplitude,params=T1Params,time=Times)
T1Fit.plot()
plt.show()
plt.plot(Times,Amplitude,'bo')
plt.show()