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
from TermColor import colored, cprint
import statsmodels.stats.stattools as stats

def TrialT2Func(time,A,T2,K):
    n = time/12.0
    #print(A * np.exp(-time/T2 - (K * time**3.0 / (n**2.0)) ))
    return A * np.exp(-time/T2 - (K * time**3.0 / (n**2.0)) )

def TwoExponential(time,A1,T2_1, A2, T2_2):
    return A1 * np.exp(-time/T2_1) + A2 * np.exp(-time/T2_2)

#cprint('AFTER EASTER DATA' + '-'*(columns -17) +'\n',color='green',on_color='on_white',attrs=['bold','dark'])
#? Light mineral oil sample 

#* Importing the LMO T2 data
LMOAEDF =  pd.read_csv("./Data/T2/LMO_AfterEaster.csv",names = ['Time','Voltage'],usecols = [3,4])

LMOAEDF = LMOAEDF[LMOAEDF['Time'] > 20e-3]

LMOAETime = np.array([ 0.10,11.80,23.80,36.00,48.00,59.95,71.70,83.70,95.60,107.80,119.70,131.70,144.0,155.5,167.5,179.6,191.9,203.9 ]) * 1e-3
LMOAEAmp = np.array( [ 5.52,4.24,3.36,2.76,2.28,1.88,1.60,1.36,1.20,1.04,0.92,0.80,0.72,0.64,0.56,0.52,0.48,0.44 ])

LMOAETime = np.array([ 11.80,23.80,36.00,48.00,59.95,71.70,83.70,95.60,107.80,119.70,131.70,144.0,155.5,167.5,179.6,191.9,203.9 ]) * 1e-3
LMOAEAmp = np.array( [ 4.24,3.36,2.76,2.28,1.88,1.60,1.36,1.20,1.04,0.92,0.80,0.72,0.64,0.56,0.52,0.48,0.44 ])

#* Plotting the Light mineral oil data
plt.figure()
plt.plot(LMOAEDF['Time'],LMOAEDF['Voltage'])
plt.plot(LMOAETime,LMOAEAmp,'rx')
plt.title('LMO After Easter T2 Data')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show(block=False)


#* Fitting the Light minearal oil data

#? Single exponential
LMOAET2Model = Model(TrialT2Func)
LMOAET2Params = Parameters()

LMOAET2Params.add('A',value=4.60,min=1e-10,vary=True)
LMOAET2Params.add('T2',value=0.073,min=1e-10,vary=True)
LMOAET2Params.add('K',value=1,min=0,vary=True)

LMOAET2Fit = LMOAET2Model.fit(LMOAEAmp,params=LMOAET2Params,time=LMOAETime,weights=1/0.02)
cprint('Light Mineral Oil after Easter, single exponential, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(colored('The Durbin-Watson test for this sample is ' + str(stats.durbin_watson(LMOAET2Fit.residual)),color='blue',attrs=['bold']))
print(LMOAET2Fit.fit_report())
LMOAET2Fit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='LMO AE T2 - Single Exponential')
plt.show(block=False)

#? Two exponentials
LMOAET2Model2 = Model(TwoExponential) 
LMOAET2Params2 = Parameters()
LMOAET2Params2.add('A1',value=4,min=1e-10,vary=True)
LMOAET2Params2.add('T2_1',value=0.01,min=1e-10,vary=True)
LMOAET2Params2.add('A2',value=5,min=1e-10,vary=True)
LMOAET2Params2.add('T2_2',value=0.1,min=1e-10,vary=True)

LMOAET2Fit2 = LMOAET2Model2.fit(LMOAEAmp,params=LMOAET2Params2,time=LMOAETime,weights=1/0.02)
cprint('LMOAE, two exponentials, all data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(colored('The Durbin-Watson test for this sample is ' + str(stats.durbin_watson(LMOAET2Fit2.residual)),color='blue',attrs=['bold']))
print(LMOAET2Fit2.fit_report())
LMOAET2Fit2.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='LMO AE T2 - Two Exponentials')
plt.show(block=False)


#! Plotting the single and double exponential functions to compare
plt.figure()
plt.plot(LMOAEDF['Time'],LMOAEDF['Voltage'],label="Data")
plt.plot(LMOAETime,LMOAEAmp,'gx')
plt.plot(LMOAEDF['Time'],TrialT2Func(LMOAEDF['Time'],LMOAET2Fit.best_values['A'],LMOAET2Fit.best_values['T2'],LMOAET2Fit.best_values['K']),label="1 Exp Fit",color='green')
plt.plot(LMOAEDF['Time'],TwoExponential(LMOAEDF['Time'],LMOAET2Fit2.best_values['A1'],LMOAET2Fit2.best_values['T2_1'],LMOAET2Fit2.best_values['A2'],LMOAET2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of single vs two exponential fit of LMOAE for T2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=False)


#* Now doing the same comparison for the first four data points and the last nine data points for the light minearal oil
#? First four

LMOAETimeFirstFour = np.array([11.80,23.80,36.00 ]) * 1e-3
LMOAEAmpFirstFour = np.array( [ 4.24,3.36,2.76])

LMOAET2FirstFourModel = Model(TrialT2Func)
LMOAET2FirstFourParams = Parameters()
LMOAET2FirstFourParams.add('A',value=4.60,min=1e-10,vary=True)
LMOAET2FirstFourParams.add('T2',value=0.073,min=1e-10,vary=True)
LMOAET2FirstFourParams.add('K',value=1,vary=True)

LMOAET2FirstFourFit = LMOAET2FirstFourModel.fit(LMOAEAmpFirstFour,params=LMOAET2FirstFourParams,time=LMOAETimeFirstFour,weights=1/0.02)
cprint('LMOAE, single exponential, first four data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(colored('The Durbin-Watson test for this sample is ' + str(stats.durbin_watson(LMOAET2FirstFourFit.residual)),color='blue',attrs=['bold']))
print(LMOAET2FirstFourFit.fit_report())
LMOAET2FirstFourFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='LMOAE T2 - First Four Single Exp')
plt.show(block=False)

#? Last nine

LMOAETimeLastNine = np.array([ 107.80,119.70,131.70,144.0,155.5,167.5,179.6,191.9,203.9  ]) * 1e-3
LMOAEAmpLastNine = np.array( [ 1.04,0.92,0.80,0.72,0.64,0.56,0.52,0.48,0.44  ])

LMOAET2LastNineModel = Model(TrialT2Func)
LMOAET2LastNineParams = Parameters()
LMOAET2LastNineParams.add('A',value=4.60,min=1e-10,vary=True)
LMOAET2LastNineParams.add('T2',value=0.073,min=1e-10,vary=True)
LMOAET2LastNineParams.add('K',value=1,min=0,vary=True)

LMOAET2LastNineFit = LMOAET2LastNineModel.fit(LMOAEAmpLastNine,params=LMOAET2LastNineParams,time=LMOAETimeLastNine,weights=1/0.02)
cprint('LMOAE, single exponential, last nine data points',color='cyan',on_color='on_magenta',attrs=['bold'])
print(colored('The Durbin-Watson test for this sample is ' + str(stats.durbin_watson(LMOAET2LastNineFit.residual)),color='blue',attrs=['bold']))
print(LMOAET2LastNineFit.fit_report())
LMOAET2LastNineFit.plot(show_init=False,xlabel='Time (s)',ylabel='Net Magnetisation',yerr=0.02,numpoints=10000,title='LMOAE T2 - Last Nine Single Exp')
plt.show(block=False)


#! Comparison plot between the three different fits for light minearal oil after easter
plt.figure()
plt.plot(LMOAEDF['Time'],LMOAEDF['Voltage'],label="Data")
plt.plot(LMOAETime,LMOAEAmp,'gx')
plt.plot(LMOAEDF['Time'],TrialT2Func(LMOAEDF['Time'],LMOAET2FirstFourFit.best_values['A'],LMOAET2FirstFourFit.best_values['T2'],LMOAET2FirstFourFit.best_values['K']),label="First Four",color='#7A0E71')
plt.plot(LMOAEDF['Time'],TrialT2Func(LMOAEDF['Time'],LMOAET2LastNineFit.best_values['A'],LMOAET2LastNineFit.best_values['T2'],LMOAET2LastNineFit.best_values['K']),label="Last Nine",color='green')
plt.plot(LMOAEDF['Time'],TrialT2Func(LMOAEDF['Time'],LMOAET2Fit.best_values['A'],LMOAET2Fit.best_values['T2'],LMOAET2Fit.best_values['K']),label="1 Exp Fit",color='orange')
plt.plot(LMOAEDF['Time'],TwoExponential(LMOAEDF['Time'],LMOAET2Fit2.best_values['A1'],LMOAET2Fit2.best_values['T2_1'],LMOAET2Fit2.best_values['A2'],LMOAET2Fit2.best_values['T2_2']),label="2 Exp Fit",color='magenta')
plt.title('Comparison of different fits for LMOAE')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show(block=True)


