# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:52:36 2024

@author: D. Michel Pino
@email: dmichel.pino@csic.es
"""

import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors

from tqdm import tqdm
from joblib import Parallel, delayed

import lib_holes as hh
import lib_artist_figures as art

import scipy.optimize as sco


psave = dict(main_path = 'figures/',
              file_format = 'png')

N = 1001

B = [0,0,0]

rainbow = ['red', 'darkorange', 'gold', 'limegreen', 'deepskyblue', 'darkorchid', 'blue', 'black']

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ############################   PANEL  1   ############################# '''

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'none'

# phi = np.pi/4
# theta = np.pi/2

# params = dict(system = '8KP-3D_'+type_Delta,
#               phi = phi,
#               theta = theta)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#              yscale = 'linear',
#              linewidth = 1.5,
#              cmap = cmap,
#              scatter = True,
#              colorbar = dict(location = 'top',
#                              aspect = 35,
#                              ticks = [-1, 1],
#                              ),
#              vmin = -1,
#              vmax = 1,
#              cbar_ticklabels = ['LH', 'HH'],
#              show = True)

# cut_ll, cut_hh = hh.gap_positions_8KP(phi, theta)

# varx = art.var('k', np.linspace(-1.2, 1.2, N)*1e9, r'$k$ $(\mathrm{nm}^{-1})$', norm = 1e9)
# varz_E = art.var('E', np.zeros((16,N)), r'$E$'+lab_Delta, norm=Enorm)
# varz_hl = art.var('comp_hl', np.zeros((16,N)), r'$|hh|^2-|lh|^2$')

# j = 0
# for k in varx.array:
#     kx = k*np.sin(theta)*np.cos(phi)
#     ky = k*np.sin(theta)*np.sin(phi)
#     kz = k*np.cos(theta)
#     varz_E.array[:,j], c_hh, c_ll, c_so, c_el = hh.band_composition_8KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH)
#     varz_hl.array[:,j] = c_hh - c_ll
#     j += 1

# pargs['ylim'] = np.array([np.min(varz_E.array), np.max(varz_E.array)])
# art.plot([varx,varz_E,varz_hl], params, pargs, psave=None, label='a)')

# varz_E.name = 'E-zoom'
# pargs['ylim'] = np.array([-1,1])*0.12
# art.plot([varx,varz_E,varz_hl], params, pargs, psave=None, label='b)')



# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# params = dict(system = '8KP-3D_'+type_Delta,
#               phi = phi,
#               theta = theta
#               )

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# pargs['ylim'] = np.array([-1,1])*0.1

# cut_ll, cut_hh = hh.gap_positions_8KP(phi, theta)

# varx = art.var('k', np.linspace(-1.2, 1.2, N)*1e9, r'$k$ $(\mathrm{nm}^{-1})$', norm = 1e9)
# varz_E = art.var('E', np.zeros((16,N)), r'$E$'+lab_Delta, norm=Enorm)
# varz_hl = art.var('comp_hl', np.zeros((16,N)), r'$|hh|^2-|lh|^2$')

# for cut in [cut_hh, cut_ll]:
#     varx.array = np.linspace(0.9998, 1.0002, N)*cut
#     j = 0
#     for k in varx.array:
#         kx = k*np.sin(theta)*np.cos(phi)
#         ky = k*np.sin(theta)*np.sin(phi)
#         kz = k*np.cos(theta)
#         varz_E.array[:,j], c_hh, c_ll, c_so, c_el = hh.band_composition_8KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH)
#         varz_hl.array[:,j] = c_hh - c_ll
#         j += 1
    
#     pargs['xticks'] = [cut*1e-9]
    
#     if abs(cut - cut_hh) < 1e-2*cut:
#         varz_hl.name = 'comp_hl_k2'
#         pargs['xticklabels'] = [r'$k_2$']
#         lab = 'd)'
#     else:
#         varz_hl.name = 'comp_hl_k1'
#         pargs['xticklabels'] = [r'$k_1$']
#         lab = 'c)'        

#     art.plot([varx,varz_E,varz_hl], params, pargs, psave=None, label=lab)







# N = 401

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               xticks = [-np.pi/2, 0, np.pi/2],
#               xticklabels = [r'$-\pi/2$', r'$0$', r'$\pi/2$'],
#               linewidth = 1.5,
#               dashes = [],
#               show = True)

# varx = art.var('phi', np.linspace(-np.pi/2, np.pi/2, N), r'$\phi_k$')
# varz1 = art.var('gap1', np.zeros((N)), r'$\Delta(k_1)$'+lab_Delta, norm=Enorm)
# varz2 = art.var('gap2', np.zeros((N)), r'$\Delta(k_2)$'+lab_Delta, norm=Enorm)

# theta = np.pi/2

# params = dict(system = '8KP-3D_'+type_Delta,
#               theta = theta
#               )

# j = 0
# for phi in tqdm(varx.array):
#     k1, k2 = hh.gap_positions_8KP(phi, theta)
#     k1 = abs(k1)
#     k2 = abs(k2)
#     kx1 = k1*np.sin(theta)*np.cos(phi)
#     ky1 = k1*np.sin(theta)*np.sin(phi)
#     kz1 = k1*np.cos(theta)
#     kx2 = k2*np.sin(theta)*np.cos(phi)
#     ky2 = k2*np.sin(theta)*np.sin(phi)
#     kz2 = k2*np.cos(theta)
#     H, H_BdG_1 = hh.Hamiltonian_8KP_3D(kx1, ky1, kz1, B, Delta_CB, Delta_LH, Delta_HH)
#     H, H_BdG_2 = hh.Hamiltonian_8KP_3D(kx2, ky2, kz2, B, Delta_CB, Delta_LH, Delta_HH)
#     eigvals_1 = np.linalg.eigh(H_BdG_1)[0]
#     eigvals_2 = np.linalg.eigh(H_BdG_2)[0]
#     varz1.array[j] = eigvals_1[8]
#     varz2.array[j] = eigvals_2[8]
#     j += 1

# pargs['ylim'] = [0.051975, 0.052825]
# pargs['color'] = 'red'
# art.plot([varx, varz1], params, pargs, psave=None, label='g)')

# pargs['ylim'] = [0.000, 0.00525]
# pargs['color'] = 'blue'
# art.plot([varx, varz2], params, pargs, psave=None, label='h)')




# params = dict(system = '8KP-3D_'+type_Delta,
#               )

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               xticks = [-np.pi/2, 0, np.pi/2],
#               xticklabels = [r'$-\pi/2$', r'$0$', r'$\pi/2$'],
#               yticks = [0, np.pi/2, np.pi],
#               yticklabels = [r'$0$', r'$\pi/2$', r'$\pi$'],
#               cmap = cmap,
#               colorbar = dict(location = 'top',
#                                           aspect = 35,
#                                           ),
#               show = True)

# N = 51

# varx = art.var('phi', np.linspace(-np.pi/2, np.pi/2, N), r'$\phi_k$')
# vary = art.var('theta', np.linspace(0, np.pi, N), r'$\theta_k$')
# varz1 = art.var('gap1', np.zeros((N,N)), r'$\Delta(k_1)$'+lab_Delta, norm=Enorm)
# varz2 = art.var('gap2', np.zeros((N,N)), r'$\Delta(k_2)$'+lab_Delta, norm=Enorm)
    
# i = 0
# for phi in tqdm(varx.array):
#     j = 0
#     for theta in vary.array:
#         k1, k2 = hh.gap_positions_8KP(phi, theta)
#         k1 = abs(k1)
#         k2 = abs(k2)
#         kx1 = k1*np.sin(theta)*np.cos(phi)
#         ky1 = k1*np.sin(theta)*np.sin(phi)
#         kz1 = k1*np.cos(theta)
#         kx2 = k2*np.sin(theta)*np.cos(phi)
#         ky2 = k2*np.sin(theta)*np.sin(phi)
#         kz2 = k2*np.cos(theta)
#         H, H_BdG_1 = hh.Hamiltonian_8KP_3D(kx1, ky1, kz1, B, Delta_CB, Delta_LH, Delta_HH)
#         H, H_BdG_2 = hh.Hamiltonian_8KP_3D(kx2, ky2, kz2, B, Delta_CB, Delta_LH, Delta_HH)
#         eigvals_1 = np.linalg.eigh(H_BdG_1)[0]
#         eigvals_2 = np.linalg.eigh(H_BdG_2)[0]
#         varz1.array[j,i] = eigvals_1[8]
#         varz2.array[j,i] = eigvals_2[8]
#         j += 1
#     i += 1

# pargs['cmap'] = 'Reds'
# pargs['colorbar']['ticks'] = [0.052, 0.0525, 0.053]
# pargs['vmin'] = np.min([np.min(varz1.array/varz1.norm), 0.052])
# pargs['vmax'] = np.max([np.max(varz1.array/varz1.norm), 0.053])
# pargs['cbar_ticklabels'] = [r'$0.052$', varz1.xlabel, r'$0.053$']

# art.plot([varx,vary,varz1], params, pargs, psave=None, label='e)')

# pargs['cmap'] = 'Blues'
# pargs['colorbar']['ticks'] = [0, 0.0025, 0.005]
# pargs['vmin'] = np.min([np.min(varz2.array/varz2.norm), 0])
# pargs['vmax'] = np.max([np.max(varz2.array/varz2.norm), 0.005])
# pargs['cbar_ticklabels'] = [r'$0$', varz2.xlabel, r'$0.005$']

# art.plot([varx,vary,varz2], params, pargs, psave=None, label='f)')





# N = 301

# params = dict(system = '8KP-3D_'+type_Delta,
#               )

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               dashes = [[],[3,3]],
#               show = True)

# varx = art.var('mu', -np.linspace(0.01, 0.25, N), r'$\mu$ (eV)')
# varz1 = art.var('gap1', np.zeros((2,N)), r'$\Delta(k_1)$'+lab_Delta, norm=Enorm)
# varz2 = art.var('gap2', np.zeros((2,N)), r'$\Delta(k_2)$'+lab_Delta, norm=Enorm)

# phi = 0
# theta = np.pi/2

# j = 0
# for mu in tqdm(varx.array):
#     k1, k2 = hh.gap_positions_8KP(phi, theta, mu)
#     k1 = abs(k1)
#     k2 = abs(k2)
#     kx1 = k1*np.sin(theta)*np.cos(phi)
#     ky1 = k1*np.sin(theta)*np.sin(phi)
#     kz1 = k1*np.cos(theta)
#     kx2 = k2*np.sin(theta)*np.cos(phi)
#     ky2 = k2*np.sin(theta)*np.sin(phi)
#     kz2 = k2*np.cos(theta)
#     H, H_BdG_1 = hh.Hamiltonian_8KP_3D(kx1, ky1, kz1, B, Delta_CB, Delta_LH, Delta_HH, mu)
#     H, H_BdG_2 = hh.Hamiltonian_8KP_3D(kx2, ky2, kz2, B, Delta_CB, Delta_LH, Delta_HH, mu)
#     eigvals_1 = np.linalg.eigh(H_BdG_1)[0]
#     eigvals_2 = np.linalg.eigh(H_BdG_2)[0]
#     varz1.array[0,j] = eigvals_1[8]
#     varz2.array[0,j] = eigvals_2[8]
#     j += 1

# phi = np.pi/4
# theta = np.pi/2

# j = 0
# for mu in tqdm(varx.array):
#     k1, k2 = hh.gap_positions_8KP(phi, theta, mu)
#     k1 = abs(k1)
#     k2 = abs(k2)
#     kx1 = k1*np.sin(theta)*np.cos(phi)
#     ky1 = k1*np.sin(theta)*np.sin(phi)
#     kz1 = k1*np.cos(theta)
#     kx2 = k2*np.sin(theta)*np.cos(phi)
#     ky2 = k2*np.sin(theta)*np.sin(phi)
#     kz2 = k2*np.cos(theta)
#     H, H_BdG_1 = hh.Hamiltonian_8KP_3D(kx1, ky1, kz1, B, Delta_CB, Delta_LH, Delta_HH, mu)
#     H, H_BdG_2 = hh.Hamiltonian_8KP_3D(kx2, ky2, kz2, B, Delta_CB, Delta_LH, Delta_HH, mu)
#     eigvals_1 = np.linalg.eigh(H_BdG_1)[0]
#     eigvals_2 = np.linalg.eigh(H_BdG_2)[0]
#     varz1.array[1,j] = eigvals_1[8]
#     varz2.array[1,j] = eigvals_2[8]
#     j += 1


# pargs['ylim'] = [-0.001, 0.07]
# pargs['color'] = 'red'
# art.plot([varx, varz1], params, pargs, psave=None)

# pargs['ylim'] = [-0.001, 0.02]
# pargs['color'] = 'blue'
# art.plot([varx, varz2], params, pargs, psave=None)





''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 2     ############################ '''

# ''' bands '''

# phi = 0
# hh.F = 0

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([-1, 1])*0.25,
#               cmap = cmap,
#               scatter = True,
#                 colorbar = dict(location = 'top',
#                                 aspect = 35,
#                                 ticks = [-1, 1],
#                                 ),
#               vmin = -1,
#               vmax = 1,
#               cbar_ticklabels = ['LH', 'HH'],
#               show = True)

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'none'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1
#     case 'disorder':
#         lab_Delta = r'$/\Delta_\mathrm{R}$'
#         disorder = [hh.Delta, 0, 0, 0]
#         params['DR'] = disorder[0]/hh.Delta
#         Enorm = hh.Delta

# varx = art.var('k', np.linspace(-1, 1, N)*1.2*1e9, r'$k$ $(\mathrm{nm}^{-1})$', norm = 1e9)
# varz_E = art.var('E', np.zeros((8,N)), r'$E$'+lab_Delta, norm=Enorm)
# varz_hl = art.var('comp_hl', np.zeros((8,N)), r'$|hh|^2-|lh|^2$')

# lab = ['a)', 'b)', 'c)']
# i = 0
# for mu in [-0.2, -0.05, -0.01]:
#     j = 0
#     for k in varx.array:
#         kx = k*np.cos(phi)
#         ky = k*np.sin(phi)
#         eigvals, c_hh, c_ll = hh.band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         varz_E.array[:,j] = eigvals
#         varz_hl.array[:,j] = c_hh - c_ll
#         j += 1
        
#     art.plot([varx,varz_E, varz_hl], params, pargs, psave=None, label=lab[i])
#     i += 1




# ''' gaps - mu '''

# N = 301

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# phi = 0

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               ylim = [0,1],
#               linewidth = 1.5,
#               cmap = cmap,
#               scatter = True,
#               colorbar = dict(location = 'top',
#                               aspect = 35,
#                               ticks = [-1, 1],
#                               ),
#               vmin = -1,
#               vmax = 1,
#               cbar_ticklabels = ['LH', 'HH'],
#               show = True)

# varx = art.var('mu', np.linspace(-0.25, 0.05, N), r'$\mu$ (eV)')
# varz = art.var('k-gaps', np.zeros((4,N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9) #+r' $\times$ $10^{-5}$'
# varz_hl = art.var('comp_hl', np.zeros((4,N)), r'$|hh|^2-|lh|^2$')
# varn = art.var('k-normal', np.zeros((N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9) #+r' $\times$ $10^{-5}$'

# hh.F = 0
# j = 0
# for mu in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(phi, mu=mu)
#     for i in range(4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = k[i]
#         eigvals, c_hh, c_ll = hh.band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         varz_hl.array[i,j] = c_hh[4] - c_ll[4]
#         if k[i]/varz.norm < 1e-2:
#             varz.array[i,j] = None
#     j += 1
    
# art.plot([varx, varz, varz_hl], params, pargs, psave=None, label='d)')



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 3     ############################ '''


''' bands and gaps '''

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               ylim = [-0.25,0.25],
#               linewidth = 1.5,
#               cmap = cmap,
#               scatter = True,
#               colorbar = dict(location = 'top',
#                               aspect = 35,
#                               ticks = [-1, 1],
#                               ),
#               vmin = -1,
#               vmax = 1,
#               cbar_ticklabels = ['LH', 'HH'],
#               show = True)

# phi = np.pi/4
# mu = -0.15
# hh.F = 5e6

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'none'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               mu = mu,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# varx = art.var('k', np.linspace(-1, 1, N)*1.2*1e9, r'$k$ $(\mathrm{nm}^{-1})$', norm = 1e9)
# varz_E = art.var('E', np.zeros((8,N)), r'$E$'+lab_Delta, norm=Enorm)
# varz_hl = art.var('comp_hl', np.zeros((8,N)), r'$|hh|^2-|lh|^2$')

# j = 0
# for k in varx.array:
#     kx = k*np.cos(phi)
#     ky = k*np.sin(phi)
#     eigvals, c_hh, c_ll = hh.band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#     varz_E.array[:,j] = eigvals
#     varz_hl.array[:,j] = c_hh - c_ll
#     j += 1
    
# art.plot([varx,varz_E, varz_hl], params, pargs, psave=None, label='a)')

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               mu = mu,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# varz_E = art.var('E', np.zeros((8,N)), r'$E$'+lab_Delta, norm=Enorm)

# cuts = hh.gap_positions_4KP_2DHG(phi, mu=mu)

# lab = ['c)', 'd)', 'e)', 'f)']
# ylim = [0.12, 0.12, 0.05, 0.05]
# xlim = [0.0005, 0.0005, 0.0001, 0.0001]
# xticklab = [r'$k_1$',r'$k_2$',r'$k_3$',r'$k_4$']

# for i in range(len(cuts)):
#     varz_E.name = 'E'+str(i+1)
#     varx.array = np.linspace(1-xlim[i], 1+xlim[i], N)*cuts[i]
#     pargs['ylim'] = np.array([-1, 1])*ylim[i]
#     pargs['xticks'] = [cuts[i]*1e-9]
#     pargs['xticklabels'] = [xticklab[i]]
#     j = 0
#     for k in varx.array:
#         kx = k*np.cos(phi)
#         ky = k*np.sin(phi)
#         varz_E.array[:,j], c_hh, c_ll = hh.band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         varz_hl.array[:,j] = c_hh - c_ll
#         j += 1
        
#     art.plot([varx,varz_E,varz_hl], params, pargs, psave=None, label=lab[i])



''' gaps - phi '''

# phi = 0
# mu = -0.15

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               # mu = mu,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# N = 301
# varx = art.var('phi', np.linspace(-np.pi/2, np.pi/2, N), r'$\phi_k$')
# varz = art.var('gaps', np.zeros((4,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)

# j = 0
# for nphi in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(nphi, mu)
#     for i in range(4):
#         kx = k[i]*np.cos(nphi)
#         ky = k[i]*np.sin(nphi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = eigvals[4]
#     j += 1

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([0, 0.12]),
#               xticks = [-np.pi/2, 0, np.pi/2],
#               xticklabels = [r'$-\pi/2$', r'$0$', r'$\pi/2$'],
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'$\Delta(k_1)$',r'$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4,
#               show = True)

# art.plot([varx,varz], params, pargs, psave=None, label='b)')


''' gaps - F '''

# phi = 0
# mu = -0.15

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               # mu = mu,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# N = 301

# varx = art.var('F', np.linspace(0, 6e6, N), r'$F$ (MV/m)', norm=1e6)
# varz = art.var('gaps', np.zeros((4,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm) #+r' $\times$ $10^{-5}$'

# j = 0
# for hh.F in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(phi, mu=mu)
#     for i in range(4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = eigvals[4]
#     j += 1

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([0, 0.12]),
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'$\Delta(k_1)$',r'$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4,
#               show = True)

# art.plot([varx, varz], params, pargs, psave=None, label='h)')


''' gaps - mu '''

# phi = 0

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'CB'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([0, 0.15]),
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'$\Delta(k_1)$',r'$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4,
#               show = True)

# N = 301

# varx = art.var('mu', np.linspace(-0.23, 0.03, N), r'$\mu$ (eV)')
# varz = art.var('gaps-malmal', np.zeros((4,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)# + r', $k_i$ $(\mathrm{nm}^{-1})$', norm=Enorm) #+r' $\times$ $10^{-5}$'
# vark = art.var('k-gaps', np.zeros((4,N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9) #+r' $\times$ $10^{-5}$'

# j = 0
# for mu in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(phi, mu=mu)
#     vark.array[:,j] = k
#     for i in range(4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = eigvals[4]
#         if vark.array[i,j]*1e-9 < 2e-5:
#             varz.array[i,j] = None
#             vark.array[i,j] = None
#     j += 1

# for i in range(4):
#     for j in range(len(varx.array)):
#         if np.isnan(vark.array[i,j]):
#             varz.array[i,j:] = None
#             vark.array[i,j:] = None
#             break

# art.plot([varx, varz], params, pargs, psave=None, label='g)')


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 4     ############################ '''

# phi = 0

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([0, 1.3]),
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'$\Delta(k_1)$',r'$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4,
#               show = True)

# N = 301


# lab = ['a)', 'b)']

# n = 0
# for type_Delta in ['HH', 'LH']:
#     params = dict(system = '4KP-2DHG_'+type_Delta,
#                   phi = phi,
#                   F = hh.F)
    
#     Delta_CB = 0
#     Delta_LH = 0
#     Delta_HH = 0
    
#     match type_Delta:
#         case 'CB':
#             lab_Delta = r'$/\Delta_\mathrm{CB}$'
#             Delta_CB = hh.Delta
#             params['DCB'] = Delta_CB/hh.Delta
#             Enorm = hh.Delta
#         case 'LH':
#             lab_Delta = r'$/\Delta_\mathrm{LH}$'
#             Delta_LH = hh.Delta
#             params['DLH'] = Delta_LH/hh.Delta
#             Enorm = hh.Delta
#         case 'HH':
#             lab_Delta = r'$/\Delta_\mathrm{HH}$'
#             Delta_HH = hh.Delta
#             params['DHH'] = Delta_HH/hh.Delta
#             Enorm = hh.Delta
#         case 'none':
#             lab_Delta = r' (eV)'
#             Delta_CB = 0
#             Delta_LH = 0
#             Delta_HH = 0
#             Enorm = 1
    
#     varx = art.var('mu', np.linspace(-0.23, 0.03, N), r'$\mu$ (eV)')
#     varz = art.var('gaps-malmal', np.zeros((4,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)# + r', $k_i$ $(\mathrm{nm}^{-1})$', norm=Enorm) #+r' $\times$ $10^{-5}$'
#     vark = art.var('k-gaps', np.zeros((4,N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9) #+r' $\times$ $10^{-5}$'
    
#     j = 0
#     for mu in tqdm(varx.array):
#         k = hh.gap_positions_4KP_2DHG(phi, mu=mu)
#         vark.array[:,j] = k
#         for i in range(4):
#             kx = k[i]*np.cos(phi)
#             ky = k[i]*np.sin(phi)
#             H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#             eigvals = np.linalg.eigh(H_BdG)[0]
#             varz.array[i,j] = eigvals[4]
#             if vark.array[i,j]*1e-9 < 2e-5:
#                 varz.array[i,j] = None
#                 vark.array[i,j] = None
#         j += 1
    
#     for i in range(4):
#         for j in range(len(varx.array)):
#             if np.isnan(vark.array[i,j]):
#                 varz.array[i,j:] = None
#                 vark.array[i,j:] = None
#                 break
    
#     art.plot([varx, varz], params, pargs, psave=None, label=lab[n])
#     n += 1



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 5     ############################ '''


''' gaps - mu '''

# phi = 0

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'disorder'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1
#     case 'disorder':
#         lab_Delta = r'$/\Delta_\mathrm{R}$'
#         disorder = [hh.Delta, 0, 0, 0]
#         params['DR'] = disorder[0]/hh.Delta
#         Enorm = disorder[0]

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = np.array([0, 1.4]),
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'$\Delta(k_1)$',r'$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4,
#               show = True)

# N = 301

# varx = art.var('mu', np.linspace(-0.23, 0.03, N), r'$\mu$ (eV)')
# varz = art.var('gaps-malmal', np.zeros((4,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)# + r', $k_i$ $(\mathrm{nm}^{-1})$', norm=Enorm) #+r' $\times$ $10^{-5}$'
# vark = art.var('k-gaps', np.zeros((4,N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9) #+r' $\times$ $10^{-5}$'

# j = 0
# for mu in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(phi, mu=mu)
#     vark.array[:,j] = k
#     for i in range(4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu, disorder=disorder)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = eigvals[4]
#         if vark.array[i,j]*1e-9 < 2e-5:
#             varz.array[i,j] = None
#             vark.array[i,j] = None
#     j += 1

# for i in range(4):
#     for j in range(len(varx.array)):
#         if np.isnan(vark.array[i,j]):
#             varz.array[i,j:] = None
#             vark.array[i,j:] = None
#             break

# art.plot([varx, varz], params, pargs, psave=None, label='a)')


''' gaps - phi '''

# phi = 0
# mu = -0.01

# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0
# type_Delta = 'disorder'

# params = dict(system = '4KP-2DHG_'+type_Delta,
#               phi = phi,
#               # mu = mu,
#               F = hh.F)

# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1
#     case 'disorder':
#         Delta_HH = 180e-6
#         Delta_R = 20e-6
#         # lab_Delta = r'$/\Delta_\mathrm{R}$'
#         lab_Delta = r'$/\Delta_0$'
#         disorder = [Delta_R, 0, 0, 0]
#         params['DHH'] = Delta_HH/hh.Delta
#         params['DR'] = disorder[0]/hh.Delta
#         Enorm = hh.Delta

# N = 301
# varx = art.var('phi', np.linspace(-np.pi, np.pi, N), r'$\phi_k$')
# varz = art.var('gaps', np.zeros((8,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)
# vark = art.var('gaps-k', np.zeros((4,N)), r'$k_i$ $(\mathrm{nm}^{-1})$', norm=1e9)

# j = 0
# for nphi in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(nphi, mu)
#     for i in range(4):
#         kx = k[i]*np.cos(nphi)
#         ky = k[i]*np.sin(nphi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu, disorder=disorder)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[i,j] = eigvals[4]
#     j += 1

# disorder[2] = np.pi/4
# j = 0
# for nphi in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG(nphi, mu)
#     for i in range(4):
#         kx = k[i]*np.cos(nphi)
#         ky = k[i]*np.sin(nphi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu, disorder=disorder)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         varz.array[4+i,j] = eigvals[4]
#     j += 1

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               ylim = [0.65, 1],
#               xticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
#               xticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
#               yticksalignment = 'top',
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]*2,
#               legend = dict(labels=[r'_$\Delta(k_1)$',r'_$\Delta(k_2)$',r'$\Delta(k_3)$',r'$\Delta(k_4)$'],
#                             loc='upper right',
#                             ncols=2),
#               dashes = [[]]*4 + [[3,3]]*4,
#               show = True)

# art.plot([varx,varz], params, pargs, psave=None, label='b)')



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 6     ############################ '''


# N = 400

# phi = 0
# mu = -0.01
# hh.F = 5e5

# type_Delta = 'HH'

# params = dict(system = '4KP-2DHG-Bfield_'+type_Delta,
#                 phi = phi,
#               mu = mu,
#               F = hh.F)
# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0        
# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1
#     case 'HH-LH':
#         lab_Delta = r'$/\Delta_\mathrm{tot}$'
#         Delta_HH = hh.Delta
#         Delta_LH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = 2*hh.Delta

''' bands '''

# phi = 0
# B = np.array([1,1,0])*1/np.sqrt(2)
# k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
# varx = art.var('k', np.linspace(0.975*k[2], 1.025*k[3], N), r'$k_x$ $(\mathrm{nm}^{-1})$', norm=1e9)
# vary = art.var('E', np.zeros((8,N)), r'$E$'+lab_Delta, norm=Enorm)
# varz = art.var('comp', np.zeros((8,N)), 'comp')

# j = 0
# for kj in varx.array:
#     kx = kj*np.cos(phi)
#     ky = kj*np.sin(phi)
#     E, comp_hh, comp_ll = hh.band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#     vary.array[:,j] = E
#     varz.array[:,j] = (comp_hh - comp_ll)
#     j += 1

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               cmap = cmap,
#                 ylim = np.array([-1,1])*4,
#                 xticks = np.array([k[2], k[3]])*1e-9,
#                 xticklabels = [r'$k_3$', r'$k_4$'],
#                 scatter = True,
#                 colorbar = dict(location = 'top',
#                                 aspect = 35,
#                                 ticks = [-1, 1],
#                                 ),
#                 cbar_ticklabels = ['LH', 'HH'],
#                 vmin = -1,
#                 vmax = 1,
#               show = True)

# art.plot([varx, vary, varz], params, pargs, psave=None, label='a)')


''' pairings and gaps - |B| '''

# N = 201

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 1.5,
#               dashes = [[]]*4+[[3,3]],
#               yticksalignment = 'bottom',
#               color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5], 'black'],
#               legend = dict(labels=['','',r'$\Delta(k_{3})$',r'$\Delta(k_{4})$', r'$\Delta^\mathrm{(gap)}(\phi_k)$'],
#                                     loc='upper right',
#                                     ncols=5),
#               show = True)

# type_Delta = 'HH'
# mu = -0.01
# hh.F = 5e5
# phi = 0

# params = dict(system = '4KP-2DHG-Bfield_'+type_Delta,
#               phi = phi,
#               mu = mu,
#               F = hh.F)
# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0        
# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1

# varx = art.var('B', np.linspace(0, 2.5, N), r'$B$ (T)')
# varz = art.var('gaps', np.zeros((5,N)), r'$\Delta(k_i)$'+lab_Delta, norm=Enorm)

# padd = dict()

# for l in range(3):
    
#     if phi == 0:
#         label = ['b)','c)','d)','e)']
#         padd['ntext'] = 1
#         padd['text'] = [dict(x = 0.92,
#                                   y = 0.04,
#                                   text = r'$k_x$',
#                                   fontsize = 18,
#                                   color = 'black',
#                                   transData = False)]
#     else:
#         label = ['d)','e)','f)']
#         padd['ntext'] = 1
#         padd['text'] = [dict(x = 0.775,
#                                   y = 0.04,
#                                   text = r'$k_x=k_y$',
#                                   fontsize = 18,
#                                   color = 'black',
#                                   transData = False)]
    
#     padd['ntext'] += 1
#     padd['text'].append(dict(x = 0.03,
#                               y = 0.04,
#                               text = '',
#                               fontsize = 18,
#                               color = 'black',
#                               transData = False))
#     match l:
#         case 0:
#             padd['text'][1]['text'] = r'$B_x$'
#             Btheta = np.pi/2
#             Bphi = 0
#         case 1:
#             padd['text'][1]['text'] = r'$B_y$'
#             Btheta = np.pi/2
#             Bphi = np.pi/2
#         case 2:
#             padd['text'][1]['text'] = r'$B_z$'
#             Btheta = 0
#             Bphi = 0
#         case 3:
#             padd['text'][1]['text'] = r'$B_x=B_y$'
#             Btheta = np.pi/2
#             Bphi = 3*np.pi/4
    
#     params['Bphi'] = Bphi
#     params['Btheta'] = Btheta
#     j = 0
#     for BI in tqdm(varx.array):
#         B = BI*np.array([np.sin(Btheta)*np.cos(Bphi), np.sin(Btheta)*np.sin(Bphi), np.cos(Btheta)])
#         k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#         E4 = np.zeros(4)
#         E3 = np.zeros(4)
#         for i in range(4):
#             kx = k[i]*np.cos(phi)
#             ky = k[i]*np.sin(phi)
#             H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#             eigvals = np.linalg.eigh(H_BdG)[0]
#             E4[i] = eigvals[4]
#             E3[i] = eigvals[3]
#         varz.array[:4,j] = (E4-E3)/2
#         varz.array[4,j] = np.min([E4,abs(E3)])
#         varz.array[:2,j] = None
#         j += 1
        
#     pargs['ylim'] = [0, 1.15]
#     art.plot([varx, varz], params, pargs, padd=padd, psave=psave, label=label[l])
    
    


''' pairings and gaps - Bphi-Btheta '''

# N = 501

# cmap = mpl.colormaps['OrRd']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               xticks = [-np.pi, 0, np.pi],
#               xticklabels = [r'$-\pi$', r'$0$', r'$\pi$'],
#               yticks = [0, np.pi/2, np.pi],
#               yticklabels = [r'$0$', r'$\pi/2$', r'$\pi$'],
#               cmap = 'OrRd',
#               colorbar = dict(location='right',
#                               aspect=50),
#               show = True)

# mu = -0.01
# hh.F = 5e5
# phi = 0

# params = dict(system = '4KP-2DHG-Bfield_'+type_Delta,
#               phi = phi,
#               mu = mu,
#               F = hh.F)

# varx = art.var('Bphi', np.linspace(-np.pi, np.pi, N), r'$\phi_B$')
# vary = art.var('Btheta', np.linspace(0, np.pi, N), r'$\theta_B$')
# varp4 = art.var('p4', np.zeros((N,N)), r'$\Delta(k_4,B=1\mathrm{T})$'+lab_Delta, norm=Enorm)
# varp3 = art.var('p3', np.zeros((N,N)), r'$\Delta(k_3,B=1\mathrm{T})$'+lab_Delta, norm=Enorm)
# varz = art.var('gaps', np.zeros((N,N)), r'$\Delta^\mathrm{gap}(\phi_k,B=1\mathrm{T})$'+lab_Delta, norm=Enorm)

# def pairings_Bphi_Btheta(i,j):
#     Bphi = varx.array[j]
#     Btheta = vary.array[i]
#     B = np.array([np.sin(Btheta)*np.cos(Bphi), np.sin(Btheta)*np.sin(Bphi), np.cos(Btheta)])
#     k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#     E4 = np.zeros(4)
#     E3 = np.zeros(4)
#     for n in range(4):
#         kx = k[n]*np.cos(phi)
#         ky = k[n]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         E4[n] = eigvals[4]
#         E3[n] = eigvals[3]
#     p3 = (E4[2]-E3[2])/2
#     p4 = (E4[3]-E3[3])/2
#     gap = np.min([E4])
#     return i, j, p3, p4, gap
# with Parallel(n_jobs=4) as parallel:
#     result = parallel(
#         delayed(pairings_Bphi_Btheta)(i,j) for i in range(N) for j in range(N)
#         )
# for i, j, p3, p4, gap in result:
#     varp3.array[i,j] = p3
#     varp4.array[i,j] = p4
#     varz.array[i,j] = gap

# pargs['vmin'] = 0.6
# pargs['vmax'] = 1
# pargs['colorbar']['ticks'] = [0.6, 0.8, 1]
# pargs['cmap'] = 'Blues'
# pargs['cbar_ticklabels'] = [r'$0.6$', varp3.xlabel, r'$1$']
# art.plot([varx, vary, varp3], params, pargs, psave=None, label='e)')
# pargs['vmin'] = 0.3
# pargs['vmax'] = 0.92
# pargs['colorbar']['ticks'] = [0.3, (0.3+0.92)/2, 0.92]
# pargs['cbar_ticklabels'] = [r'$0.3$', varz.xlabel, r'$0.92$']
# pargs['cmap'] = 'Reds'
# art.plot([varx, vary, varz], params, pargs, psave=None, label='f)')



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 8     ############################ '''

# type_Delta = 'HH'

# mu = -0.01
# hh.F = 5e5

# params = dict(system = '4KP-2DHG-Bfield_'+type_Delta,
#               mu = mu,
#               F = hh.F)
# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0        
# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1


''' bands - phi_k '''

# N = 100

# B = [0.07,0,0]
# params['B'] = B
# varx = art.var('phik', np.linspace(0, 2*np.pi, N), r'$\phi_k$')
# vary = art.var('bandas', np.zeros((8,N)), r'$E$'+lab_Delta, norm=Enorm)

# j = 0
# for phi in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#     for i in range(2,4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         vary.array[(i-2),j] = eigvals[3]
#         vary.array[i,j] = eigvals[4]
#     j += 1

# B = [0,0,0]
# j = 0
# for phi in tqdm(varx.array):
#     k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#     for i in range(2,4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         vary.array[4+(i-2),j] = eigvals[3]
#         vary.array[4+i,j] = eigvals[4]
#     j += 1

# cmap = mpl.colormaps['Spectral']
# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#                aspect = 3/4/2,
#               linewidth = 1.5,
#               color = [rainbow[4], rainbow[5], rainbow[4], rainbow[5]] + ['black']*4,
#               dashes = [[3,3]]*2 + [[]]*2 + [[1,1]]*4,
#               alpha = [1]*4 + [0.25]*4,
#               ylim = np.array([0.85,1.1]),
#               xaxistop = True,
#               yticksalignment='bottom',
#               xticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]),
#               xticklabels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
#               legend = dict(labels=['','',r'$\Delta(k_{3})$',r'$\Delta(k_{4})$'],
#                                                     loc='upper right',
#                                                     ncols=1),
#               show = True)

# padd = dict(ncurves=0,
#             curves = [],
#             nbackgrounds=0,
#             background = [],
#             ntext=0,
#             text = [],
#             narrows=0,
#             arrows = []
#             )
# for phi in np.array([1/4, 2/4, 3/4, 4/4, 5/4, 6/4, 7/4])*np.pi:
#     padd['ncurves'] += 1
#     padd['curves'].append(dict(x=phi*np.ones(2),
#                           y = pargs['ylim'],
#                           color='black',
#                           linewidth=1,
#                           dashes=[3,3],
#                           alpha=0.5,
#                           transData=True))

# art.plot([varx, vary], params, pargs, padd=padd, psave=None, label='a)')

''' bands - low Bx '''

# N = 5

# phi_array = np.array([1/4, 2/4, 4/4, 5/4, 6/4])*np.pi
# L = len(phi_array)
# k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
# varx = art.var('Bx', np.linspace(0, 0.25, N), r'$B_x$ (T)')
# vary = art.var('banditas3-kk', np.zeros((L,N)), r'$\Delta(k_3)$'+lab_Delta, norm=Enorm)
# vary2 = art.var('banditas4-kk', np.zeros((L,N)), r'$\Delta(k_4)$'+lab_Delta, norm=Enorm)

# n = 0
# for phi in phi_array:
#     j = 0
#     for Bx in tqdm(varx.array):
#         B = [Bx, 0, 0]
#         k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#         i = 2
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         vary.array[n,j] = eigvals[4]
        
#         i = 3
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
#         vary2.array[n,j] = eigvals[4]
#         j += 1
#     n += 1

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 2,
#               color = ['dodgerblue','dodgerblue', 'blue','blue', 'darkblue'],
#               dashes = [[2,5]]*5,#[[3,3],[],[],[3,3],[]],
#               legend = dict(labels=[r'$\phi_k=1/4\pi, 3/4\pi$',r'$\phi_k=1/2\pi$',r'$\phi_k=0, \pi, 2\pi$',r'$\phi_k=5/4\pi, 7/4\pi$', r'$\phi_k=2/3\pi$'],
#                                                                     loc='upper center',
#                                                                     bbox_to_anchor =(0.5,1.4),
#                                                                     ncols=3),
#               ylim = np.array([0.85,1.1]),
#               show = True)
# art.plot([varx, vary], params, pargs, psave=None, label='b)')

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               linewidth = 2,
#               color = ['violet','violet', 'magenta','magenta', 'mediumvioletred'],
#               dashes = [[2,5]]*5,
#               ylim = np.array([0.85,1.1]),
#               legend = dict(labels=[r'$\phi_k=1/4\pi, 3/4\pi$',r'$\phi_k=1/2\pi$',r'$\phi_k=0, \pi, 2\pi$',r'$\phi_k=5/4\pi, 7/4\pi$', r'$\phi_k=2/3\pi$'],
#                                                                     loc='upper center',
#                                                                     bbox_to_anchor =(0.5,1.4),
#                                                                     ncols=3),
#               show = True)
# art.plot([varx, vary2], params, pargs, psave=None, label='c)')


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' ###########################     PANEL 9     ############################ '''

# type_Delta = 'HH'

# mu = -0.1
# hh.F = 5e5

# params = dict(system = '4KP-2DHG-Bfield_'+type_Delta,
#               mu = mu,
#               F = hh.F)
# Delta_CB = 0
# Delta_LH = 0
# Delta_HH = 0        
# match type_Delta:
#     case 'CB':
#         lab_Delta = r'$/\Delta_\mathrm{CB}$'
#         Delta_CB = hh.Delta
#         params['DCB'] = Delta_CB/hh.Delta
#         Enorm = hh.Delta
#     case 'LH':
#         lab_Delta = r'$/\Delta_\mathrm{LH}$'
#         Delta_LH = hh.Delta
#         params['DLH'] = Delta_LH/hh.Delta
#         Enorm = hh.Delta
#     case 'HH':
#         lab_Delta = r'$/\Delta_\mathrm{HH}$'
#         Delta_HH = hh.Delta
#         params['DHH'] = Delta_HH/hh.Delta
#         Enorm = hh.Delta
#     case 'none':
#         lab_Delta = r' (eV)'
#         Delta_CB = 0
#         Delta_LH = 0
#         Delta_HH = 0
#         Enorm = 1


# ''' BOGOBANDS '''


# N = 101

# varphi = art.var('phi', np.linspace(0, 1, N)*np.pi*2, r'$\phi$ (rad)')
# varx = art.var('kx', np.array([]), r'$k_x$ $(\mathrm{nm}^{-1})$', norm=1e9)
# vary = art.var('ky', np.array([]), r'$k_y$ $(\mathrm{nm}^{-1})$', norm=1e9)
# varz = art.var('bogobands-malmal', np.array([]), '')
# I = 1.5
# Bphi = 0
# B = I*np.array([np.cos(Bphi),np.sin(Bphi),0])
# params['B'] = list(B)

# def func(k, phi, i, B, Delta_CB, Delta_LH, Delta_HH, mu):
#     kx = k*np.cos(phi)
#     ky = k*np.sin(phi)
#     H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#     eigvals = np.linalg.eigh(H_BdG)[0]
#     return eigvals[i]

# j = 0
# for phi in tqdm(varphi.array):
#     k = hh.gap_positions_4KP_2DHG_finite_B(phi, B, mu=mu)
#     ku = np.array([(k[0]+k[1])/2, k[1]*1.1, (k[2]+k[3])/2, k[3]*1.1])
#     kd = np.array([k[0]*0.9, (k[0]+k[1])/2, k[2]*0.9, (k[2]+k[3])/2])
#     for i in range(4):
#         kx = k[i]*np.cos(phi)
#         ky = k[i]*np.sin(phi)
#         H, H_BdG = hh.Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=mu)
#         eigvals = np.linalg.eigh(H_BdG)[0]
        
#         if eigvals[3] >= 0:
#             ki = sco.brentq(func, k[i], ku[i], args=(phi, 3, B, Delta_CB, Delta_LH, Delta_HH, mu))
#             varx.array = np.append(varx.array, ki*np.cos(phi))
#             vary.array = np.append(vary.array, ki*np.sin(phi))
#             varz.array = np.append(varz.array, i/3)
#             ki = sco.brentq(func, kd[i], k[i], args=(phi, 3, B, Delta_CB, Delta_LH, Delta_HH, mu))
#             varx.array = np.append(varx.array, ki*np.cos(phi))
#             vary.array = np.append(vary.array, ki*np.sin(phi))
#             varz.array = np.append(varz.array, i/3)
#         if eigvals[4] <= 0:
#             ki = sco.brentq(func, k[i], ku[i], args=(phi, 4, B, Delta_CB, Delta_LH, Delta_HH, mu))
#             varx.array = np.append(varx.array, ki*np.cos(phi))
#             vary.array = np.append(vary.array, ki*np.sin(phi))
#             varz.array = np.append(varz.array, i/3)
#             ki = sco.brentq(func, kd[i], k[i], args=(phi, 4, B, Delta_CB, Delta_LH, Delta_HH, mu))
#             varx.array = np.append(varx.array, ki*np.cos(phi))
#             vary.array = np.append(vary.array, ki*np.sin(phi))
#             varz.array = np.append(varz.array, i/3)

# color = [rainbow[0], rainbow[2], rainbow[4], rainbow[5]]
# cmap_name = 'my_list'
# cmap = colors.LinearSegmentedColormap.from_list(cmap_name, color, N=4)

# pargs = dict(xscale = 'linear',
#               yscale = 'linear',
#               aspect = 4/4,
#                 xlim = np.array([-1, 1])*0.75,
#                 ylim = np.array([-1, 1])*0.75,
#                 linewidth = 0.1,
#               cmap = cmap,
#               vmin = 0,
#               vmax = 1,
#                 scatter = True,
#               show = True)

# art.plot([varx,vary,varz], params, pargs, psave=None)


