import nonScatMWRadTran

import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy 

from datetime import datetime
from netCDF4 import Dataset


################################################################################
# This function computes the absolute humidity from the mixing ratio
################################################################################
def w2a(w, p, T):
    '''
    specific to absolute humidty
    '''
    Rair = 287.04  # J/kg/K
    Rvapor = 461.5  # J/kg/K
    
    q = w / (1. + w)
        
    rho = p / (Rair * T * (1 + (Rvapor / Rair - 1) * q))  # density kg/m3
    return q*rho

################################################################################
# This function computes the mixing ratio of water from the relative humidity
################################################################################.

def rh2w(temp, rh, pres):
    e = epres(temp, rh)        # Get the vapor pressure
    e = e /100.                # Convert the vapor pressure to mb (same as pres)
    w = 0.622 * e / (pres - e) # this ratio is in g/g
    w = w * 1000               # convert to g/kg
    
    return w
    
################################################################################
# This funciton computes the dew point given temperature and relative humidity
################################################################################

def rh2dpt(temp,rh):
    es0 = 611.0
    gascon = 461.5
    trplpt = 273.16
    tzero = 273.15
    
    yo = np.where(np.array(temp) == 0)[0]
    if len(yo) > 0:
        return np.zeros(len(temp))

    latent = 2.5e6 - 2.386e3*temp
    dpt = np.copy(temp)
    
    for i in range(2):
        latdew = 2.5e6 - 2.386e3*dpt
        dpt = 1.0 / ((latent/latdew) * (1.0 / (temp + tzero) - gascon/latent * np.log(rh)) + 1.0 / trplpt * (1.0 - (latent/latdew))) - tzero
    
    return dpt
    
    
################################################################################
# This function computes the vapor pressure give the temperature and relative
# humidity
################################################################################

def epres(temp, rh):
    ep = rh * esat(temp,0)
    return ep
    
################################################################################
# This function computes the saturation vapor pressure over liquid water or ice
################################################################################

def esat(temp,ice):
    es0 = 611.0
    gascon = 461.5
    trplpt = 273.16
    tzero = 273.15
    
    # Compute saturation vapor pressure (es, in mb) over water or ice at temperature
    # temp (in Kelvin using te Goff-Gratch formulation (List, 1963)
    #print(type(temp))
    if ((type(temp) != np.ndarray) & (type(temp) != list) & (type(temp) != np.ma.MaskedArray)):
        temp = np.asarray([temp])

    if type(temp) == list:
        temp = np.asarray(temp)

    tk = temp + tzero
    es = np.zeros(len(temp))
  
    if ice == 0:
        wdx = np.arange(len(temp))
        nw = len(temp)
        nice = 0
    else:
        icedx = np.where(tk <= 273.16)[0]
        wdx = np.where(tk > 267.16)[0]
        nw = len(wdx)
        nice = len(icedx)
    
    if nw > 0:
        y = 373.16/tk[wdx]
        es[wdx] = (-7.90298 * (y - 1.0) + 5.02808 * np.log10(y) -
            1.3816e-7 * (10**(11.344 * (1.0 - (1.0/y))) - 1.0) +
            8.1328e-3 * (10**(-3.49149 * (y - 1.0)) - 1.0) + np.log10(1013.246))
            
    if nice > 0:
        # for ice
        y = 273.16/tk[icedx]
        es[icedx] = (-9.09718 * (y - 1.0) - 3.56654 * np.log10(y) +
                    0.876793 * (1.0 - (1.0/y)) + np.log10(6.1071))
    
    es = 10.0**es
    
    # convert from millibar (mb) to Pa
    es = es * 100
    return es

def corr_plot(data, heights):
    """
    A handy funtion to create a correlation plot for temperature and mixing ratio
    """
    
    nz = len(heights)
    x, y = np.meshgrid(heights, heights)
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(10)
    fig.set_figwidth(5)
    print(np.max(data[:nz, :nz]))
    m1 = ax1.contourf(x, y, data[:nz, :nz], levels=np.arange(-1, 1.05, .05), cmap=cm.balance, vmin=-1, vmax=1)
    ax1.set_title("Temperature Correlation")
    fig.colorbar(m1, ax=ax1)
     
    m2 = ax2.contourf(x, y, data[nz:2*nz, nz:2*nz], levels=np.arange(-1, 1.05, .05), cmap=cm.balance, vmin=-1, vmax=1)

    ax2.set_title("Moisture Correlation")
    fig.colorbar(m2, ax=ax2)
    return fig
    
    
def cov2corr(A):
    """
    Utility function to convert a covariance matrix to a correlation matrix
    """
    d = np.sqrt(A.diagonal())
    return ((A.T/d).T)/d


def forwardRT(X,               # State vector [T(z), q(z)]
               z,               # Heights agl [m]
               pressure=None,   # Pressure [hPa]
               zenithAngle=0,   # Zenith angle of the mwr measurements [deg]
               frequencies=[]): # Frequencies to return
    """
    Converts a temperature/humidity profile into brightness temperatures at the
    given requencies
    """
    
    k = len(z)
    
    # X contains T and Q, lets split the vector
    temperature, humidity = (X[0:k]+273.15, X[k:k*2])
    
    # Convert to kg/kg
    humidity = humidity.copy() * 1e-3
        
    # Convert pressure from hPa to Pa
    pressure = pressure.copy() * 100
        
    assert np.all(np.diff(z) > 0)

    # get absolute humidity from specific humdity
    abs_humidity = w2a(humidity, pressure, temperature)
    
    # run the forward operator
    TB, tau, tau_wv, tau_o2 = nonScatMWRadTran.STP_IM10(
        z,  # [m]  
        temperature,  # [K]
        pressure,  # [Pa]
        abs_humidity,  # [kgm^-3]
        zenithAngle,  # zenith angle of observation in deg.
        frequencies,  # frequency vector in GHz
    )
    return TB

def compute_jacobian_finite_diff(Xn,    # State vecotor [T(z), q(z)]
                                 z,     # Height agl [m]
                                 p,     # Pressure [hPa]
                                 freq): # Frequencies to use in the forward model
    
    """
    Computes the jacobian through finite differences
    """
    
    
    k = len(z)
    t = np.copy(Xn[0:k])          # degC
    w = np.copy(Xn[k:2*k])        # g/kg
    
    # Allocate space for the Jacobian and forward calculation
    Kij = np.zeros((len(freq),len(Xn)))
    FXn = np.zeros(len(freq))
    
    # Perform the baseline run
    a = forwardRT(Xn, z, p, frequencies=freq)
    FXn = a.copy()
    
    # Perturb the temperature
    delta = 1.0
    for kk in range(k):
        t0 = np.copy(t)
        t0[kk] += delta
        b = forwardRT(np.append(t0, w), z, p, frequencies=freq)
        FXp = np.copy(b)
        Kij[:, kk] = (FXp-FXn) / (t0[kk] - t[kk])
        
    # Perturb the moisture
    delta = .99 #  Multiplicative perturbation of 1%
    for kk in range(k):
        w0 = np.copy(w) 
        w0[kk] *= delta
        b = forwardRT(np.append(t, w0), z, p, frequencies=freq)
        FXp = np.copy(b)
        Kij[:, kk+k] = (FXp-FXn) / (w0[kk] - w[kk])
        
    return Kij, FXn

def do_mwroe_retrieval(Xa,         # Prior
                       Sa,         # Prior covariance
                       Y,          # Observations
                       Sy,         # Observation uncertainty
                       freqs,      # Frequencies in Y
                       pressure,   # Pressure [hPa]
                       heights,    # Height agl [m]
                       max_iter=5):# Max number of iterations

    Xn = np.copy(Xa) 
    gfac = 1
    Fxnm1 = np.array([-999.])

    for i in range(max_iter):

        SaInv = scipy.linalg.pinv(Sa)
        SmInv = scipy.linalg.pinv(Sy)
        Kij, FXn = compute_jacobian_finite_diff(Xn, heights, pressure, freqs)

        B = (gfac * SaInv) + Kij.T.dot(SmInv).dot(Kij)
        Binv = scipy.linalg.pinv(B)
        Gain = Binv.dot(Kij.T).dot(SmInv)
        Xnp1 = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        Sop = Binv.dot(gfac*gfac*SaInv + Kij.T.dot(SmInv).dot(Kij)).dot(Binv)
        SopInv = scipy.linalg.pinv(Sop)
        Akern = Binv.dot(Kij.T).dot(SmInv).dot(Kij)

        if len(Fxnm1) == len(Y):
            di2m = ((FXn[:,None] - Fxnm1[:,None]).T.dot(
                    scipy.linalg.pinv(Kij.dot(Sop).dot(Kij.T)+Sy)).dot(
                    FXn[:,None] - Fxnm1[:,None]))[0,0]
        else:
            di2m = 9.0e9

        if di2m < len(Y):
            print("Converged!")
            # This will return the optimal state vector, the posterior covariance matrix, and the forward calculation 
            return Xn, Sop, FXn

        Xn = np.squeeze(Xnp1).copy()
        Fxnm1 = FXn.copy()
        rmsa = np.sqrt(np.sum(((Y - FXn)/np.diag(Sy))**2) / len(Y))

        print('iter is ' + str(i) + ' di2m is ' + str(di2m) + ' and RMS is ' + str(rmsa))
        
    print("Did not converge...")
    return Xn, Sop, FXn