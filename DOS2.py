# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:27:34 2024

@author: D. Michel Pino
@email: dmichel.pino@csic.es
"""

import lib_holes as hh
import lib_artist_figures as art
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

''' 
    This file contains the functions for calculating the density of states
    of the (supeconducting) 4KP 2DHG model for a varying magnetic field
'''


# Parameters:

# chemical potential
mu = -0.01

# Induced pairings (via CB, LH or HH)

type_Delta = 'HH'

Delta_CB = 0
Delta_LH = 0
Delta_HH = 0
match type_Delta:
    case 'CB':
        lab_Delta = r'$/\Delta_\mathrm{CB}$'
        Delta_CB = hh.Delta
        Enorm = hh.Delta
    case 'LH':
        lab_Delta = r'$/\Delta_\mathrm{LH}$'
        Delta_LH = hh.Delta
        Enorm = hh.Delta
    case 'HH':
        lab_Delta = r'$/\Delta_\mathrm{HH}$'
        Delta_HH = hh.Delta
        Enorm = hh.Delta
    case 'none':
        lab_Delta = r' (eV)'
        Delta_CB = 0
        Delta_LH = 0
        Delta_HH = 0
        Enorm = 1

# Size of Bfield space
K = 32
# Size of omega space
M = 32
# Size of k-phi space
N = 256

# Direction of the Bfield in spherical coordinates
Bphi = 0
Btheta = np.pi/2

# Declaration of x (Bfield), y (omega) and z (DOS) axis
varx = art.var('B', np.linspace(0.8, 2, K), r'$|B|$ (T), $\phi_B/\pi={Bphi}$, $\theta_B/\pi={Btheta}$'.format(Bphi=Bphi/np.pi, Btheta=Btheta/np.pi))
vary = art.var('omega', np.linspace(-0.5, 0.5, M)*Enorm, r'$E$'+lab_Delta, norm=Enorm)
varz = art.var('DOS', np.zeros((M,K)), r'$\rho(E)/\rho_\mathrm{max}$')

# Declaration of phi axis
phi_array = np.linspace(0, 2*np.pi, N)
# Declaration of |k| axis for integration (to be used in units of k_3 and k_4)
kabs_array = np.linspace(0.99, 1.01, N)
# Spectral broadening
delta = 1e-3*hh.Delta

# Calculation of k_3 and k_4 for each phi and B
def parallelize_gap_positions(i,j):
    phi = phi_array[j]
    B = [varx.array[i],0,0]
    B = varx.array[i] * np.array([np.sin(Btheta)*np.cos(Bphi), np.sin(Btheta)*np.sin(Bphi), np.cos(Btheta)])
    k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
    return i, j, k[2], k[3]

with Parallel(n_jobs=16) as parallel:
    result = parallel(
        delayed(parallelize_gap_positions)(i,j) for i in range(K) for j in range(N)
        )

# Declaration of matrices to store the results
k3_array = np.zeros((K,N))
k4_array = np.zeros((K,N))
# Fill the matrices with the parallelized results
for i, j, v3, v4 in result:
    k3_array[i,j] = v3
    k4_array[i,j] = v4

# Calculation of the energy levels [E_m, E_p] (E_m = -E_p) of lowest (absolute) energy around k_3 and k_4
def parallelize_spectrum(i,j):
    # For each phi and B
    phi = phi_array[j]
    # Select a different orientation of the Bfield if you want
    B = [varx.array[i],0,0]
    # Declaration of energy arrays
    Ep3 = np.zeros(N)
    Ep4 = np.zeros(N)
    Em3 = np.zeros(N)
    Em4 = np.zeros(N)
    n = 0
    # In a |k| window around each gap position k_3 and k_4, we calculate these energy levels
    for kabs in kabs_array:
        kj = k3_array[i,j]*kabs
        kx = kj*np.cos(phi)
        ky = kj*np.sin(phi)
        H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        Ep3[n] = eigvals[4]
        Em3[n] = eigvals[3]
    
        kj = k4_array[i,j]*kabs
        kx = kj*np.cos(phi)
        ky = kj*np.sin(phi)
        H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        Ep4[n] = eigvals[4]
        Em4[n] = eigvals[3]
        n += 1
    return i, j, Ep3, Em3, Ep4, Em4

with Parallel(n_jobs=16) as parallel:
    result = parallel(
        delayed(parallelize_spectrum)(i,j) for i in range(K) for j in range(N)
        )
# Declaration of matrices to store the results
Ep3 = np.zeros((K,N,N))
Ep4 = np.zeros((K,N,N))
Em3 = np.zeros((K,N,N))
Em4 = np.zeros((K,N,N))
# Fill the matrices with the parallelized results
for i, j, vp3, vm3, vp4, vm4 in result:
    Ep3[i,j,:] = vp3
    Em3[i,j,:] = vm3
    Ep4[i,j,:] = vp4
    Em4[i,j,:] = vm4

def parallelize_DOS(o):
    # For each omega
    omega = vary.array[o]
    # Imaginary part of Green's function for each eigenenergy-array (negative -m- and positive -p- levels around k_3 and k_4)
    ImGp3 = delta/np.pi / ((omega-Ep3)**2 + delta**2)
    ImGm3 = delta/np.pi / ((omega-Em3)**2 + delta**2)
    ImGp4 = delta/np.pi / ((omega-Ep4)**2 + delta**2)
    ImGm4 = delta/np.pi / ((omega-Em4)**2 + delta**2)
    # k-integrated DOS (separated for negative and positive branches)
    Ip = np.zeros((K,N))
    Im = np.zeros((K,N))
    # k-phi-integrated DOS (separated for negative and positive branches)
    DOSp = np.zeros(K)
    DOSm = np.zeros(K)
    for i in range(K):
        # For each phi, we integrate ImG in k around k_3 and k_4
        for j in range(N):
            Ip[i,j] = np.trapz(ImGp3[i,j]*kabs_array*k3_array[i,j], x=kabs_array*k3_array[i,j]) + np.trapz(ImGp4[i,j]*kabs_array*k4_array[i,j], x=kabs_array*k4_array[i,j])
            Im[i,j] = np.trapz(ImGm3[i,j]*kabs_array*k3_array[i,j], x=kabs_array*k3_array[i,j]) + np.trapz(ImGm4[i,j]*kabs_array*k4_array[i,j], x=kabs_array*k4_array[i,j])
        # For each B, we integrate I in phi
        DOSp[i] = np.trapz(Ip[i], x=phi_array)
        DOSm[i] = np.trapz(Im[i], x=phi_array)
    return o, DOSm, DOSp

with Parallel(n_jobs=16) as parallel:
        result = parallel(
            delayed(parallelize_DOS)(o) for o in tqdm(range(M), desc="Progress")
            )
# Declaration of matrices to store the results
DOSm = np.zeros((M,K))
DOSp = np.zeros((M,K))
# Fill the matrices with the parallelized results
for o, vm, vp in result:
    DOSm[o,:] = vm
    DOSp[o,:] = vp

''' 
    Now, we need to renormalize the DOS(omega) curves for each B.
    Since DOS is calculated in arbitrary units, this renormalization will not
    change the functional form of the DOS(omega). This renormalization is
    needed because numerical integration will give different values of rho_max
    for each B. To present these results properly, we will try to renormalize
    the DOS such as it behaves smoothly as a function of B.
    
    The simplest renormalization is just to renormalize all the DOS(omega) to 
    DOS_max=1, giving a good result.
'''

for i in range(K):
    DOSm[:,i] *= 1/np.max(DOSm[:,i])
    DOSp[:,i] *= 1/np.max(DOSp[:,i])
varz.array = DOSm + DOSp
varz.norm = np.max(varz.array)

''' PLOTS '''

# Dictionary of system parameters
params = dict(system = '4KP-2DHG-DOS_'+type_Delta,
              mu = mu,
              F = hh.F,
              DCB = Delta_CB/hh.Delta,
              DLH = Delta_LH/hh.Delta,
              DHH = Delta_HH/hh.Delta)

# Dictionary of plot parameters
pargs = dict(xscale = 'linear',
             yscale = 'linear',
             cmap = 'inferno',
             vmin = 0,
             vmax = 1,
             colorbar = dict(location='right',
                             aspect=50,
                             ticks=[0, 1/2, 1]),
             cbar_ticklabels = [r'$0$', varz.xlabel, r'$1$'],
             show = True)

# Dictionary of extra elements in the plot
padd = dict(ncurves=0,
            curves = [],
            nbackgrounds=0,
            ntext=0,
            text = [],
            narrows=0,
            arrows = [],
            )

# Dictionary of saving parameters
psave = dict(main_path = 'figures/',
              file_format = 'png')

art.plot([varx,vary,varz], params, pargs, padd=padd, psave=None)


