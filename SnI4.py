import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fsweep,fsweep_signal =\
        np.loadtxt('fsweep.csv',\
           delimiter=',',skiprows=1,unpack=True)
t90,t90_signal =\
        np.loadtxt('t90.csv',\
           delimiter=',',skiprows=1,unpack=True)
T1,T1_signal =\
        np.loadtxt('T1.csv',\
           delimiter=',',skiprows=1,unpack=True)
T2,T2_signal =\
        np.loadtxt('T2.csv',\
           delimiter=',',skiprows=1,unpack=True)
zeeman,zeeman_signal =\
        np.loadtxt('zeeman.csv',\
           delimiter=',',skiprows=1,unpack=True)

# fsweep fit (Gaussian)
##def gauss_fit(freq,A,f0,y0,width):
##    gauss = y0 + A*np.exp(-((freq-f0)**2)/(2*(width**2)))
##    return gauss
##p0 = np.array([16,409.3,0,.005])
##coeffs, matcov = curve_fit(gauss_fit,fsweep,fsweep_signal,p0)
##fsweep_fit = gauss_fit(fsweep,coeffs[0],\
##                       coeffs[1], coeffs[2],\
##                       coeffs[3])
##print coeffs
##try:    
##    plt.plot(fsweep,fsweep_signal,'x',fsweep,fsweep_fit,'k-')
##    plt.title('Zero-field NQR Resonance of tin(IV) iodide')
##    plt.xlabel('RF Frequency (MHz)')
##    plt.ylabel('Signal (AU)')
##    plt.text(409.35,10,'f0 = %s MHz\n\nwidth = %s MHz'%\
##             (round(coeffs[1],3),round(coeffs[3],4)))
##    plt.show()
##except KeyboardInterrupt:
##    plt.close()


# t90
##try:    
##    plt.plot(t90*(10**6),t90_signal,'x')
##    plt.title('t90 Nutation of tin(IV) iodide')
##    plt.xlabel('Pulse length (us)')
##    plt.ylabel('Signal (AU)')
##    plt.show()
##except KeyboardInterrupt:
##    plt.close()


# T1
def T1_fit(M,F,T1,a1,a2):
    T1_func = 
try:    
    plt.semilogx(T1*(10**6),T1_signal,'x')
    plt.title('T1 Nutation of tin(IV) iodide')
    plt.xlabel('Pulse length (us)')
    plt.ylabel('Signal (AU)')
    plt.show()
except KeyboardInterrupt:
    plt.close()























