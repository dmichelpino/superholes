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
    of the (supeconducting) 4KP 2DHG model for a given magnetic field
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

# Size of omega space
M = 128
# Size of k-phi space
N = 512

# Declaration of x (omega) and y (DOS) axis
varx = art.var('omega', np.linspace(-1.1, 1.1, M)*Enorm, r'$E$'+lab_Delta, norm=Enorm)
vary = art.var('DOS-cut', np.zeros((M)), r'$\rho(E)/\rho_\mathrm{max}$')

# Declaration of phi axis
phi_array = np.linspace(0, 2*np.pi, N)
# Declaration of |k| axis for integration (to be used in units of k_3 and k_4)
kabs_array = np.linspace(0.99, 1.01, N)
# Spectral broadening
delta = 1e-3*hh.Delta
# Magnetic field [Bx, By, Bz] (T)
B = [1.5, 0, 0]

# Calculation of k_3 and k_4 for each phi
def parallelize_gap_positions(phi):
    k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
    return k[2], k[3]

with Parallel(n_jobs=16) as parallel:
        result = parallel(
            delayed(parallelize_gap_positions)(phi) for phi in tqdm(phi_array, desc="Progress")
            )
k3_array = np.array(result)[:,0]
k4_array = np.array(result)[:,1]

# Calculation of the energy levels [E_m, E_p] (E_m = -E_p) of lowest (absolute) energy around k_3 and k_4
def parallelize_spectrum(i):
    # For each phi
    phi = phi_array[i]
    # Declaration of energy arrays
    Ep3 = np.zeros(N)
    Ep4 = np.zeros(N)
    Em3 = np.zeros(N)
    Em4 = np.zeros(N)
    n = 0
    # In a |k| window around each gap position k_3 and k_4, we calculate these energy levels
    for kabs in kabs_array:
        kj = k3_array[i]*kabs
        kx = kj*np.cos(phi)
        ky = kj*np.sin(phi)
        H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        Ep3[n] = eigvals[4]
        Em3[n] = eigvals[3]
    
        kj = k4_array[i]*kabs
        kx = kj*np.cos(phi)
        ky = kj*np.sin(phi)
        H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        Ep4[n] = eigvals[4]
        Em4[n] = eigvals[3]
        n += 1
    return i, Ep3, Em3, Ep4, Em4

with Parallel(n_jobs=24) as parallel:
        result = parallel(
            delayed(parallelize_spectrum)(i) for i in tqdm(range(N), desc="Progress")
            )
# Declaration of matrices to store the results
Ep3 = np.zeros((N,N))
Ep4 = np.zeros((N,N))
Em3 = np.zeros((N,N))
Em4 = np.zeros((N,N))
# Fill the matrices with the parallelized results
for i, vp3, vm3, vp4, vm4 in result:
    Ep3[i,:] = vp3
    Em3[i,:] = vm3
    Ep4[i,:] = vp4
    Em4[i,:] = vm4
    
def parallelize_DOS(o):
    # For each omega
    omega = varx.array[o]
    # Imaginary part of Green's function for each eigenenergy-array (negative -m- and positive -p- levels around k_3 and k_4)
    ImGp3 = delta/np.pi / ((omega-Ep3)**2 + delta**2)
    ImGm3 = delta/np.pi / ((omega-Em3)**2 + delta**2)
    ImGp4 = delta/np.pi / ((omega-Ep4)**2 + delta**2)
    ImGm4 = delta/np.pi / ((omega-Em4)**2 + delta**2)
    # k-integrated DOS (separated for negative and positive branches)
    Ip = np.zeros((N))
    Im = np.zeros((N))
    # For each phi, we integrate ImG in k around k_3 and k_4
    for j in range(N):
        Ip[j] = np.trapz(ImGp3[j]*kabs_array*k3_array[j], x=kabs_array*k3_array[j]) + np.trapz(ImGp4[j]*kabs_array*k4_array[j], x=kabs_array*k4_array[j])
        Im[j] = np.trapz(ImGm3[j]*kabs_array*k3_array[j], x=kabs_array*k3_array[j]) + np.trapz(ImGm4[j]*kabs_array*k4_array[j], x=kabs_array*k4_array[j])
    # k-phi-integrated DOS (separated for negative and positive branches)
    DOSp = np.trapz(Ip, x=phi_array)
    DOSm = np.trapz(Im, x=phi_array)
    return o, DOSm, DOSp

with Parallel(n_jobs=24) as parallel:
        result = parallel(
            delayed(parallelize_DOS)(o) for o in tqdm(range(M), desc="Progress")
            )
# Declaration of matrices to store the results
DOSm = np.zeros(M)
DOSp = np.zeros(M)
# Fill the matrices with the parallelized results
for o, dm, dp in result:
    DOSm[o] = dm
    DOSp[o] = dp
vary.array = DOSm + DOSp
vary.norm = np.max(vary.array)

''' PLOTS '''

# Dictionary of system parameters
params = dict(system = '4KP-2DHG-DOS_'+type_Delta,
              B = B,
              mu = mu,
              F = hh.F,
              DCB = Delta_CB/hh.Delta,
              DLH = Delta_LH/hh.Delta,
              DHH = Delta_HH/hh.Delta)

# Dictionary of plot parameters
pargs = dict(xscale = 'linear',
             yscale = 'linear',
             color = 'royalblue',
             linewidth = 1.5,
             dashes = [],
             ylim = [0, np.max(vary.array/vary.norm)*1.1],
             show = True)

# Dictionary of extra elements in the plot
padd = dict(ncurves=0,
            curves = [],
            nbackgrounds=0,
            ntext=0,
            text = [],
            narrows=0,
            arrows = [],
            area_under_curve = dict(ncurves = 1,
                                    color = 'deepskyblue',
                                    alpha = 0.2)
            )

# Dictionary of saving parameters
psave = dict(main_path = 'figures/',
              file_format = 'png')

art.plot([varx,vary], params, pargs, padd=padd, psave=None)


