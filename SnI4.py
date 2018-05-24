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
##def T1_fit(t,M0,c,a1,a2,T1a,T1b):
##    T1_func = M0*(1-c*(a1*np.exp(-(t/T1a)\
##                  + a2*np.exp(-t/T1b))))
##    return T1_func
##T1_test_fit = T1_fit(T1,11,.2,1,1,.0001,.001)
##p0 = np.array([11,.2,1,1,.0001,.001])
##coeffs, matcov = curve_fit(T1_fit,T1,T1_signal,p0)
##T1_fit = T1_fit(T1,coeffs[0],\
##                coeffs[1], coeffs[2],\
##                coeffs[3],coeffs[4],\
##                coeffs[5])
##print coeffs
##try:    
##    plt.semilogx(T1,T1_signal,'x',label="Inversion recovery sequence data")
##    plt.semilogx(T1,T1_fit,'k',label=r"$T_1$ fit")
##    plt.title('Inversion recovery of $SnI_4$ 3/2<->5/2 resonance')
##    plt.xlabel(r'$t_{wait}$ (s)')
##    plt.ylabel('Signal (AU)')
##    plt.text(1e-4,6.5,\
##    r'$M_z(t_{wait}) = M_0\left[1-\alpha\left(ae^{-t_{wait}/T_{1a}}+be^{-t_{wait}/T_{1b}}\right)\right]$')
##    plt.text(1e-4,6,r'a = 0.586, $T_{1a} = 185\mu s$')
##    plt.text(1e-4,5.5,r'b = 2.121, $T_{1b} = 127\mu s$')
##    plt.legend(loc='center right')
##    plt.show()
##except KeyboardInterrupt:
##    plt.close()



# T2
##def T2_fit(tau,Mxy,T2,y0):
##    T2_func = Mxy*np.exp(-2*tau/T2)+y0
##    return T2_func
##T2_test_fit = T2_fit(T2,14,.00015,.3)
##p0 = np.array([14,.00015,.3])
##coeffs, matcov = curve_fit(T2_fit,T2,T2_signal,p0)
##print coeffs
##T2_fit = T2_fit(T2,coeffs[0],coeffs[1],coeffs[2])
##try:
##    plt.plot(T2*10**6,T2_signal,'xb',label="Hahn echo sequence data")
##    plt.plot(T2*10**6,T2_fit,'k',label=r'$T_2$ Fit')
##    plt.legend(loc='upper right')
##    plt.title(r'Hahn echo data and $T_2$ fitting for $SnI_4$ 3/2<->5/2 resonance')
##    plt.xlabel(r'$\tau$ (us)')
##    plt.ylabel('Hahn echo signal')
##    plt.text(150,5,r'$T_2 = 107\mu s$')
##    plt.text(150,6,r'$M_{xy}(t) = M_{xy}(0)e^{-2\tau/T_2}$')
##    plt.show()
##except KeyboardInterrupt:
##    plt.close()




















