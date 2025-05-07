# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:13:52 2024

@author: D. Michel Pino
@email: dmichel.pino@csic.es
"""

import numpy as np
from scipy.optimize import fsolve

# Definition of 1/2 Pauli matrices
sigma0 = np.array([[1,0],[0,1]])
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])

# Kronecker product of spin (s) and particle-hole (t) Pauli matrices: sitj = \sigma_i \otimes \tau_j
s0t0 = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])
s0ty = np.array([[0,0,-1j,0],
                 [0,0,0,-1j],
                 [1j,0,0,0],
                 [0,1j,0,0]])
s0tz = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,-1,0],
                 [0,0,0,-1]])
sxt0 = np.array([[0,1,0,0],
                 [1,0,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])
sxtz = np.array([[0,1,0,0],
                 [1,0,0,0],
                 [0,0,0,-1],
                 [0,0,-1,0]])
syt0 = np.array([[0,-1j,0,0],
                 [1j,0,0,0],
                 [0,0,0,-1j],
                 [0,0,1j,0]])
syty = np.array([[0,0,0,-1],
                 [0,0,1,0],
                 [0,1,0,0],
                 [-1,0,0,0]])
sytz = np.array([[0,-1j,0,0],
                 [1j,0,0,0],
                 [0,0,0,1j],
                 [0,0,-1j,0]])
szt0 = np.array([[1,0,0,0],
                 [0,-1,0,0],
                 [0,0,1,0],
                 [0,0,0,-1]])
sztx = np.array([[0,0,1,0],
                 [0,0,0,-1],
                 [1,0,0,0],
                 [0,-1,0,0]])
sztz = np.array([[1,0,0,0],
                 [0,-1,0,0],
                 [0,0,-1,0],
                 [0,0,0,1]])

# Definition of 3/2 pauli matrices
Jx = 1/2*np.array([[0, np.sqrt(3), 0, 0],
                   [np.sqrt(3), 0, 2, 0],
                   [0, 2, 0, np.sqrt(3)],
                   [0, 0, np.sqrt(3), 0]])
Jy = 1j/2*np.array([[0,-np.sqrt(3), 0, 0],
                    [np.sqrt(3), 0,-2, 0],
                    [0, 2, 0,-np.sqrt(3)],
                    [0, 0, np.sqrt(3), 0]])
Jz = 1/2*np.array([[3, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0,-1, 0],
                   [0, 0, 0,-3]])
Jx3 = 1/8*np.array([[0, 7*np.sqrt(3), 0, 6],
                    [7*np.sqrt(3), 0, 20, 0],
                    [0, 20, 0, 7*np.sqrt(3)],
                    [6, 0, 7*np.sqrt(3), 0]])
Jy3 = 1j/8*np.array([[0, -7*np.sqrt(3), 0, 6],
                     [7*np.sqrt(3), 0, -20, 0],
                     [0, 20, 0, -7*np.sqrt(3)],
                     [-6, 0, 7*np.sqrt(3), 0]])
Jz3 = 1/8*np.array([[27, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -27]])

# Definition of useful matrices for magnetic-field terms in KL model (Winkler's book)
Ux = 1/(3*np.sqrt(2))*np.array([[-np.sqrt(3), 0],
                                [0, -1],
                                [1, 0],
                                [0, np.sqrt(3)]])
Uy = 1j/(3*np.sqrt(2))*np.array([[np.sqrt(3), 0],
                                 [0, 1],
                                 [1, 0],
                                 [0, np.sqrt(3)]])
Uz = np.sqrt(2)/3*np.array([[0, 0],
                            [1, 0],
                            [0, 1],
                            [0, 0]])
Txy = 1j/np.sqrt(6)*np.array([[0, 0, 0, -1],
                              [-1, 0, 0, 0]])
Tyz = 1j/(2*np.sqrt(6))*np.array([[-1, 0, -np.sqrt(3), 0],
                                  [0, np.sqrt(3), 0, -1]])
Tzx = 1/(2*np.sqrt(6))*np.array([[-1, 0, np.sqrt(3), 0],
                                 [0, np.sqrt(3), 0, -1]])

# System parameters

# Vertical magnetic field F
F = 5e6

# Universal constants
hbar = 6.582119569 * 1e-16
muB = 5.7883818060 * 1e-5
m0 = 5.68572 * 1e-12

# Superconducting gap
Delta = 200 * 1e-6

# KL parameters
Eg = 0.8981
Eso = 0.289
gamma1 = 13.37
gamma2 = 4.23
gamma3 = 5.68
kappa = 3.41
q = 0.06
P_GE = 9.19 * 1e-10 # eV.m, Kane energy

# KL-like effective superconducting parameter
gamma1D = 7.38

''' HAMILTONIANS '''

def Hamiltonian_8KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    ''' 8KP bulk Kane model. Including CB, LH, HH, SO bands '''
    px = hbar*kx
    py = hbar*ky
    pz = hbar*kz
    [Bx, By, Bz] = B
    
    m_ast = m0*0.041
    m_prime = m0 / (m0/m_ast - 2/3* 2*m0/hbar**2 * P_GE**2/Eg - 1/3* 2*m0/hbar**2 * P_GE**2/(Eg+Eso))
    
    gamma1_8KP = gamma1 - 1/3* 2*m0/hbar**2 * P_GE**2/Eg
    gamma2_8KP = gamma2 - 1/6* 2*m0/hbar**2 * P_GE**2/Eg
    gamma3_8KP = gamma3 - 1/6* 2*m0/hbar**2 * P_GE**2/Eg
    
    g_prime = 2
    kappa_prime = kappa - 1/6* 2*m0/hbar**2 * P_GE**2/Eg
    q_prime = q
    D_prime = 0
    
    H_cv6 = ( P_GE/hbar * np.array([[-1/np.sqrt(2)*(px+1j*py), np.sqrt(2/3)*pz, 1/np.sqrt(6)*(px-1j*py), 0, -1/np.sqrt(3)*pz, -1/np.sqrt(3)*(px-1j*py)],
                                    [0, -1/np.sqrt(6)*(px+1j*py), np.sqrt(2/3)*pz, 1/np.sqrt(2)*(px-1j*py), -1/np.sqrt(3)*(px+1j*py), 1/np.sqrt(3)*pz]])
             + np.block([1j/np.sqrt(3)*muB*D_prime * (Tyz*Bx + Txy*Bz + Tzx*By), np.zeros((2,2))]) )
    
    H_c = ( -mu*np.eye(2)
           + np.array([[Eg + (px**2 + py**2 + pz**2)/(2*m_prime), 0],
                       [0, Eg + (px**2 + py**2 + pz**2)/(2*m_prime)]])
           + muB/2 * g_prime * (sigmax*Bx + sigmay*By + sigmaz*Bz) )
    
    P = gamma1_8KP/(2*m0)*(px**2 + py**2 + pz**2)
    Q = gamma2_8KP/(2*m0)*(px**2 + py**2 - 2*pz**2)
    R = np.sqrt(3)/(2*m0) * (-gamma2_8KP*(px**2 - py**2) + 2j*gamma3_8KP*((px*py)))
    S = np.sqrt(3)/(m0) * gamma3_8KP*((px-1j*py)*pz)
    
    
    
    H_6KP = ( -mu*np.eye(6)
             - np.array([[P+Q, -S, R, 0, -1/np.sqrt(2)*S, np.sqrt(2)*R],
                         [-np.conj(S), P-Q, 0, R, -np.sqrt(2)*Q, np.sqrt(3/2)*S],
                         [np.conj(R), 0, P-Q, S, np.sqrt(3/2)*np.conj(S), np.sqrt(2)*Q],
                         [0, np.conj(R), np.conj(S), P+Q, -np.sqrt(2)*np.conj(R), -1/np.sqrt(2)*np.conj(S)],
                         [-1/np.sqrt(2)*np.conj(S), -np.sqrt(2)*Q, np.sqrt(3/2)*S, -np.sqrt(2)*R, P + Eso, 0],
                         [np.sqrt(2)*np.conj(R), np.sqrt(3/2)*np.conj(S), np.sqrt(2)*Q, -1/np.sqrt(2)*S, 0, P + Eso]])
             + np.block([[-2*muB*(kappa_prime*(Jx*Bx + Jy*By + Jz*Bz) + q_prime*(Jx3*Bx + Jy3*By + Jz3*Bz)), -3*muB*kappa_prime*(Ux*Bx + Uy*By + Uz*Bz)],
                         [np.conj((-3*muB*kappa_prime*(Ux*Bx + Uy*By + Uz*Bz)).T), -2*muB*kappa_prime*(sigmax*Bx + sigmay*By + sigmaz*Bz)]]) )
    
    
    H_Delta_CB = Delta_CB*np.array([[0,1],[-1,0]])
    H_Delta_LH = Delta_LH * np.array([[0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, -1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]])
    H_Delta_HH = Delta_HH * np.array([[0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [-1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]])
    
    H_8KP = np.block([[H_c,              H_cv6],
                      [np.conj(H_cv6.T), H_6KP]])
    
    H_BdG = np.block([[H_c,                     H_cv6,                                  H_Delta_CB,         np.zeros((2,6))],
                      [np.conj(H_cv6.T),        H_6KP,                                  np.zeros((6,2)),    H_Delta_LH + H_Delta_HH],
                      [np.conj(H_Delta_CB.T),   np.zeros((2,6)),                        -np.conj(H_c),      -np.conj(H_cv6)],
                      [np.zeros((6,2)),         np.conj((H_Delta_LH + H_Delta_HH).T),   -H_cv6.T,           -np.conj(H_6KP)]])
    
    return H_8KP, H_BdG

def Hamiltonian_6KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    ''' 6KP bulk Kane model. Including LH, HH, SO bands '''
    px = hbar*kx
    py = hbar*ky
    pz = hbar*kz
    [Bx, By, Bz] = B
    
    P = gamma1/(2*m0)*(px**2 + py**2 + pz**2)
    Q = gamma2/(2*m0)*(px**2 + py**2 - 2*pz**2)
    R = np.sqrt(3)/(2*m0) * (-gamma2*(px**2 - py**2) + 2j*gamma3*((px*py)))
    S = np.sqrt(3)/(m0) * gamma3*((px-1j*py)*pz)
    
    H_4KP = ( -mu*np.eye(4) - np.array([[P+Q, -S, R, 0],
                                        [-np.conj(S), P-Q, 0, R],
                                        [np.conj(R), 0, P-Q, S],
                                        [0, np.conj(R), np.conj(S), P+Q]]) )
    
    H_vso = ( -mu*np.eye(2) - np.array([[P + Eso, 0],
                                        [0, P + Eso]]) )
    
    H_v_vso = - np.array([[-1/np.sqrt(2)*S, np.sqrt(2)*R],
                [-np.sqrt(2)*Q, np.sqrt(3/2)*S],
                [np.sqrt(3/2)*np.conj(S), np.sqrt(2)*Q],
                [-np.sqrt(2)*np.conj(R), -1/np.sqrt(2)*np.conj(S)]])
    
    H_6KP = ( np.block([[H_4KP,               H_v_vso],
                        [np.conj(H_v_vso.T),  H_vso]])
            + np.block([[-2*muB*(kappa*(Jx*Bx + Jy*By + Jz*Bz) + q*(Jx3*Bx + Jy3*By + Jz3*Bz)), -3*muB*kappa*(Ux*Bx + Uy*By + Uz*Bz)],
                        [np.conj((-3*muB*kappa*(Ux*Bx + Uy*By + Uz*Bz)).T), -2*muB*kappa*(sigmax*Bx + sigmay*By + sigmaz*Bz)]]) )
    
    PD = -gamma1D/(2*m0) * (px**2 + py**2 + pz**2)
    QD = -gamma1D/(4*m0) * (px**2 + py**2 - 2*pz**2)
    RD = np.sqrt(3)/(4*m0) * gamma1D * (px**2 - py**2 - 2j*px*py)
    SD = -np.sqrt(3)/(2*m0) * gamma1D * (px - 1j*py)*pz
    LD = 0
    
    H_Delta_CB = Delta_CB/(Eg - 2*mu) * ( np.array([[0, -RD, -SD, -PD-QD],
                                                    [RD, -LD, PD-QD, np.conj(SD)],
                                                    [SD, -PD+QD, np.conj(LD), -np.conj(RD)],
                                                    [PD+QD, -np.conj(SD), np.conj(RD), 0]]) )
    H_Delta_LH = Delta_LH * np.array([[0, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 0]])
    H_Delta_HH = Delta_HH * np.array([[0, 0, 0, 1],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [-1, 0, 0, 0]])
        
    H_Delta_vso = Delta_CB * Eg / ((Eg+Eso)*(Eg-2*mu-Eso)) * np.array([[0, -PD],
                                                                             [PD, 0]])
    
    H_Delta_v_vso = 1/2 * Delta_CB * (1/(Eg-2*mu) + Eg/((Eg+Eso)*(Eg-2*mu-Eso))) * (-np.sqrt(2)) * np.array([[RD, SD/2],
                                                                                                                 [np.sqrt(3)*SD/2, QD],
                                                                                                                 [QD, -np.sqrt(3)*np.conj(SD)/2],
                                                                                                                 [-np.conj(SD)/2, np.conj(RD)]])
    
    H_Delta = np.block([[H_Delta_CB + H_Delta_LH + H_Delta_HH,  H_Delta_v_vso],
                        [-H_Delta_v_vso.T,                      H_Delta_vso]])
    
    H_BdG = np.block([[H_6KP,               H_Delta],
                      [np.conj(H_Delta.T),  -np.conj(H_6KP)]])
    
    return H_6KP, H_BdG
    
    
def Hamiltonian_4KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    ''' 4KP bulk Kane model. Including LH, HH bands '''
    px = hbar*kx
    py = hbar*ky
    pz = hbar*kz
    [Bx, By, Bz] = B
    
    P = gamma1/(2*m0)*(px**2 + py**2 + pz**2)
    Q = gamma2/(2*m0)*(px**2 + py**2 - 2*pz**2)
    R = np.sqrt(3)/(2*m0) * (-gamma2*(px**2 - py**2) + 2j*gamma3*((px*py)))
    S = np.sqrt(3)/(m0) * gamma3*((px-1j*py)*pz)
    
    H_4KP = ( -mu*np.eye(4) - np.array([[P+Q, -S, R, 0],
                                        [-np.conj(S), P-Q, 0, R],
                                        [np.conj(R), 0, P-Q, S],
                                        [0, np.conj(R), np.conj(S), P+Q]])
              - 2*muB*(kappa*(Jx*Bx + Jy*By + Jz*Bz) + q*(Jx3*Bx + Jy3*By + Jz3*Bz)) )
    
    PD = -gamma1D/(2*m0) * (px**2 + py**2 + pz**2)
    QD = -gamma1D/(4*m0) * (px**2 + py**2 - 2*pz**2)
    RD = np.sqrt(3)/(4*m0) * gamma1D * (px**2 - py**2 - 2j*px*py)
    SD = -np.sqrt(3)/(2*m0) * gamma1D * (px - 1j*py)*pz
    LD = 0
    
    H_Delta_CB = Delta_CB/(Eg - 2*mu) * ( np.array([[0, -RD, -SD, -PD-QD],
                                                    [RD, -LD, PD-QD, np.conj(SD)],
                                                    [SD, -PD+QD, np.conj(LD), -np.conj(RD)],
                                                    [PD+QD, -np.conj(SD), np.conj(RD), 0]]) )
    H_Delta_LH = Delta_LH * np.array([[0, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 0]])
    H_Delta_HH = Delta_HH * np.array([[0, 0, 0, 1],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [-1, 0, 0, 0]])
    
    H_BdG = np.block([[H_4KP,                                               H_Delta_CB + H_Delta_LH + H_Delta_HH],
                      [np.conj((H_Delta_CB + H_Delta_LH + H_Delta_HH).T),   -np.conj(H_4KP)]])
    
    return H_4KP, H_BdG

def Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1, R0=None, disorder=[0,0,0,0]):
    ''' 4KP 2DHG Kane model. Including LH, HH bands '''
    px = hbar*kx
    py = hbar*ky
    [Bx, By, Bz] = B
    
    Lw = 16*1e-9 # m
    bv = -2.16 # eV
    av = 0
    eps_parl = -0.61*1e-2
    eps_perp = 0.45*1e-2
    
    m_perp_H = m0/(gamma1-2*gamma2)
    m_perp_L = m0/(gamma1+2*gamma2)
    bH, bL = find_beta_HL(F, Lw, m_perp_H, m_perp_L)
        
    # EH_QW = hbar**2*(np.pi**2 + bH**2) / (2*m_perp_H*Lw**2)
    EH_strain = bv*(eps_perp - eps_parl) - av*(eps_perp + 2*eps_parl)
    
    # EL0_QW = hbar**2*(np.pi**2 + bL**2) / (2*m_perp_L*Lw**2)
    EL0_strain = bv*(eps_parl - eps_perp) - av*(eps_perp + 2*eps_parl)
    
    EH = EH_strain #EH_QW + 
    EL0 = EL0_strain #EL0_QW + 
    
    K = np.sqrt(bH*bL*(np.pi**2 + bH**2)*(np.pi**2 + bL**2)*(1/np.tanh(bH)-1)*(1/np.tanh(bL)-1))
    bbar = bH + bL
    pz2H = hbar**2/Lw**2 * (np.pi**2 + bH**2)
    pz2L = hbar**2/Lw**2 * (np.pi**2 + bL**2)
    # O0 = 4 * (np.exp(bbar)-1) * K / (bbar*(4*np.pi**2 + bbar**2))
    alpha0 = 2*hbar * (bL - bH) * (np.exp(bbar)-1) * K / (Lw*bbar*(4*np.pi**2 + bbar**2))
    # z0 = 8*Lw*K/(4*np.pi**2 + bbar**2) * ((np.exp(bbar)-1)/(4*np.pi**2+bbar**2) - (bbar+2+np.exp(bbar)*(bbar-2))/(4*bbar**2))
    pzHL = 1j*alpha0
    
    PH = gamma1/(2*m0)*(px**2 + py**2 + pz2H) + EH
    QH = gamma2/(2*m0)*(px**2 + py**2 - 2*pz2H)
    PL = gamma1/(2*m0)*(px**2 + py**2 + pz2L)  + EL0
    QL = gamma2/(2*m0)*(px**2 + py**2 - 2*pz2L)
    R = np.sqrt(3)/(2*m0) * (-gamma2*(px**2 - py**2) + 2j*gamma3*((px*py)))
    S = np.sqrt(3)/(m0) * gamma3*((px-1j*py)*pzHL)
        
    if R0 != None:
        R = 0
        
    H_4KP = ( -mu*np.eye(4) - np.array([[PH+QH,         -S,         R,              0],
                                        [-np.conj(S),   PL-QL,      0,              R],
                                        [np.conj(R),    0,          PL-QL,          -S],
                                        [0,             np.conj(R), -np.conj(S),    PH+QH]])
               - 2*muB*(kappa*(Jx*Bx + Jy*By + Jz*Bz) + q*(Jx3*Bx + Jy3*By + Jz3*Bz)) )
    
    H_4KP_c = ( +mu*np.eye(4) + np.array([[PH+QH,  np.conj(S),     np.conj(R),  0],
                                        [S,        PL-QL,          0,           np.conj(R)],
                                        [R,        0,              PL-QL,       np.conj(S)],
                                        [0,        R,              S,           PH+QH]])
               + 2*muB*(kappa*(Jx*Bx + np.conj(Jy)*By + Jz*Bz) + q*(Jx3*Bx + np.conj(Jy3)*By + Jz3*Bz)) )
    
    PDH = -gamma1D/(2*m0) * (px**2 + py**2 + pz2H)
    QDH = -gamma1D/(4*m0) * (px**2 + py**2 - 2*pz2H)
    PDL = -gamma1D/(2*m0) * (px**2 + py**2 + pz2L)
    QDL = -gamma1D/(4*m0) * (px**2 + py**2 - 2*pz2L)
    
    RD = np.sqrt(3)/(4*m0) * gamma1D * (px**2 - py**2 - 2j*px*py)
    SD = -np.sqrt(3)/(2*m0) * gamma1D * (px - 1j*py)*pzHL
    LD = 0
    
    H_Delta_CB = Delta_CB/(Eg - 2*mu) * ( np.array([[0,         -RD,            -SD,            -PDH - QDH],
                                                    [RD,        -LD,            PDL - QDL,      -np.conj(SD)],
                                                    [-SD,       -PDL + QDL,     np.conj(LD),    -np.conj(RD)],
                                                    [PDH + QDH, -np.conj(SD),   np.conj(RD),    0]]) )
    H_Delta_LH = Delta_LH * np.array([[0, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 0]])
    H_Delta_HH = Delta_HH * np.array([[0, 0, 0, 1],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [-1, 0, 0, 0]])
    [DR, DS, chiR, chiS] = disorder
    H_Delta_disorder = np.array([[0,                    DR*np.exp(1j*chiR),     DS*np.exp(1j*chiS),     0],
                                 [-DR*np.exp(1j*chiR),  0,                      0,                      -DS*np.exp(-1j*chiS)],
                                 [-DS*np.exp(1j*chiS),  0,                      0,                      DR*np.exp(-1j*chiR)],
                                 [0,                    DS*np.exp(-1j*chiS),    -DR*np.exp(-1j*chiR),   0]])
    H_Delta = H_Delta_CB + H_Delta_LH + H_Delta_HH + H_Delta_disorder
    
    H_BdG = np.block([[H_4KP,                H_Delta],
                      [np.conj(H_Delta.T),   H_4KP_c]])
    
    return H_4KP, H_BdG

def find_beta_HL(F, Lw, m_perp_H, m_perp_L):
    ''' Function that minimizes beta parameter from integrating vertical motion '''
    def func(beta, F, Lw, m_perp):
        dE0 = -F*Lw*(np.pi**4+3*beta[0]**4)/(2*beta[0]**2*(np.pi**2+beta[0]**2)**2) + beta[0]*hbar**2/(Lw**2*m_perp) + 1/2*F*Lw*1/np.sinh(beta[0])**2
        return dE0
    
    bH = fsolve(func, 1, args=(F, Lw, m_perp_H))
    bL = fsolve(func, 1, args=(F, Lw, m_perp_L))
    return bH[0], bL[0]

def Hamiltonian_2KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, b4, mu=-0.1):
    ''' 2KP 2DHG Kane model. Including HH band '''
    px = hbar*kx
    py = hbar*ky
    [Bx, By, Bz] = B

    Lw = 16*1e-9 # m
    bv = -2.16 # eV
    av = 0
    eps_parl = -0.61*1e-2
    eps_perp = 0.45*1e-2
    
    m_perp_H = m0/(gamma1-2*gamma2)
    m_perp_L = m0/(gamma1+2*gamma2)
    bH, bL = find_beta_HL(F, Lw, m_perp_H, m_perp_L)
        
    EH_QW = hbar**2*(np.pi**2 + bH**2) / (2*m_perp_H*Lw**2)
    EH_strain = bv*(eps_perp - eps_parl) - av*(eps_perp + 2*eps_parl)
    
    EL0_QW = hbar**2*(np.pi**2 + bL**2) / (2*m_perp_L*Lw**2)
    EL0_strain = bv*(eps_parl - eps_perp) - av*(eps_perp + 2*eps_parl)
    
    EH = EH_QW + EH_strain
    EL0 = EL0_QW + EL0_strain
    EHL = EL0 - EH
    
    K = np.sqrt(bH*bL*(np.pi**2 + bH**2)*(np.pi**2 + bL**2)*(1/np.tanh(bH)-1)*(1/np.tanh(bL)-1))
    bbar = bH + bL
    pz2 = hbar**2/Lw**2 * (np.pi**2 + bH**2)
    O0 = 4 * (np.exp(bbar)-1) * K / (bbar*(4*np.pi**2 + bbar**2))
    alpha0 = 2*hbar * (bL - bH) * (np.exp(bbar)-1) * K / (Lw*bbar*(4*np.pi**2 + bbar**2))
    z0 = 8*Lw*K/(4*np.pi**2 + bbar**2) * ((np.exp(bbar)-1)/(4*np.pi**2+bbar**2) - (bbar+2+np.exp(bbar)*(bbar-2))/(4*bbar**2))
    
    # bH = ( 2*F**2*m_perp_H**2/(9*np.pi*hbar**4) )**(1/3)
    # bL = ( 2*F**2*m_perp_L**2/(9*np.pi*hbar**4) )**(1/3)
    
    # EH_F = 3*bH*hbar**2*(gamma1-2*gamma2)/(2*m0)
    # EL0_F = 3*bL*hbar**2*(gamma1+2*gamma2)/(2*m0)
    # EH_strain = -bv*(eps_parl - eps_perp) # - av*(2*eps_parl + eps_perp)
    # EL0_strain = -bv*(-eps_parl + eps_perp) # - av*(2*eps_parl + eps_perp)
    
    # EH = EH_F + EH_strain
    # EL0 = EL0_F + EL0_strain
    # EHL = EL0 - EH + 2*np.pi**2*hbar**2*gamma2/(m0*Lw**2)
    
    # pz2 = 3*bH*hbar**2
    # O0 = 2*np.sqrt(2)*(bH*bL)**(3/4)/(bH + bL)**(3/2)
    # alpha0 = 4*hbar*np.sqrt(2/np.pi)*(bH*bL)**(3/4)*(bL - bH)/(bH + bL)**2
    # z0 = 4*np.sqrt(2/np.pi)*(bH*bL)**(3/4)/(bH + bL)**2
    # # b0 = 2*hbar*np.sqrt(2)*(bH*bL)**(3/4)*(2*bL - bH)/(bH + bL)**(5/2)
    
    gamma_h0 = 6*alpha0**2*gamma3**2/(EHL*m0)
    m_par = m0/(gamma1 + gamma2) - m0/gamma_h0
    alpha_circ = 3*alpha0*gamma3*(gamma2 + gamma3)*O0/(2*EHL*m0**2)
    alpha_square = 3*alpha0*gamma3*(gamma2 - gamma3)*O0/(2*EHL*m0**2)

    lamb = 6*(4*gamma3**2*alpha0*z0 + hbar*gamma2*kappa*O0**2)/(hbar*m0*EHL)
    lamb_prime = 6*gamma2*(4*gamma3*alpha0*z0 + hbar*kappa*O0**2)/(hbar*m0*EHL)
    # lamb_xy = 6*gamma3*(2*(gamma2+gamma3)*alpha0*z0 + hbar*kappa*O0**2)
    
    a_plus = alpha_square + alpha_circ
    a_minus = alpha_square - 3*alpha_circ

    gxx = lamb*px**2 - lamb_prime*py**2 + 3*q
    gyy = lamb_prime*px**2 - lamb*py**2 - 3*q
    gzz = 6*kappa + 27/2*q - 2*gamma_h0
    
    H0 = -mu*sigma0 - ( 1/(2*m_par)*(px**2 + py**2)*sigma0 + muB/2*(Bx*gxx*sigmax + By*gyy*sigmay + Bz*gzz*sigmaz)
          + a_plus*px**3*sigmay + a_minus*px**2*py*sigmax + a_minus*px*py**2*sigmay + a_plus*py**3*sigmax + b4*(kx**2 + ky**2)**2*sigma0)
    
    PD = -gamma1D/(2*m0) * (px**2 + py**2 + pz2)
    QD = -gamma1D/(4*m0) * (px**2 + py**2 - 2*pz2)
    
    pm = px - 1j*py
    pp = px + 1j*py
    Delta_HH_0 = 1j*(gamma2-gamma3)*O0/4*(pm**2*pp + pm*pp**2) + 1j*(gamma2+5*gamma3)*O0/4*(pm**3 + pp**3)
    Delta_HH_1 = 0
    Delta_HH_2 = -2j*alpha0*gamma3*(px**2 + py**2)
    Delta_HH_3 = -1j*(gamma2-gamma3)*O0/4*(pm**2*pp - pm*pp**2) + 1j*(gamma2+5*gamma3)*O0/4*(pm**3 - pp**3)
    H_Delta_CB = -Delta_CB/(Eg - 2*(mu+EH)) * (PD+QD) * 1j*sigmay + ( 3*Delta_CB*EL0*gamma1D*alpha0 )/( 4*(Eg-2*(mu+EH))*(-EHL-2*(mu+EH))*EHL*m0**2 ) * (Delta_HH_0*sigma0 + Delta_HH_1*sigmax + Delta_HH_2*sigmay + Delta_HH_3*sigmaz)
    H_BdG = np.block([[H0,                      H_Delta_CB],
                      [np.conj(H_Delta_CB.T),   -np.conj(H0)]])
    
    return H0, H_BdG

def Hamiltonian_InAs_6KP_3D(kx, ky, kz, mu=-0.1):
    ''' 6KP bulk Kane model for InAs. Including CB, LH, HH bands '''
    px = hbar*kx
    py = hbar*ky
    pz = hbar*kz
    
    ### InAs parameters (Governale's paper) ###
    Eg = 0.418 # eV
    P_GE = 9.9197 * 1e-10 # eV.m
    m_ast = m0*0.0229
    gamma1 = 20.4
    gamma2 = 8.3
    gamma3 = 9.1
    mu = -0.1 # eV
    Delta_CB = 1e-3
    # Delta_CB = 0
    
    gamma1_6KP = gamma1# - 1/3* 2*m0/hbar**2 * P_GE**2/Eg
    gamma2_6KP = gamma2# - 1/6* 2*m0/hbar**2 * P_GE**2/Eg
    gamma3_6KP = gamma3# - 1/6* 2*m0/hbar**2 * P_GE**2/Eg

    # m_prime = m0 / (m0/m_ast - 2/3* 2*m0/hbar**2 * P_GE**2/Eg - 1/3* 2*m0/hbar**2 * P_GE**2/(Eg+Delta0))
    m_prime = m_ast
    
    H_c = ( -mu*np.eye(2)
            + np.array([[Eg + (px**2 + py**2 + pz**2)/(2*m_prime), 0],
                        [0, Eg + (px**2 + py**2 + pz**2)/(2*m_prime)]]) )
    
    H_cv4 = ( P_GE/hbar * np.array([[-1/np.sqrt(2)*(px+1j*py), np.sqrt(2/3)*pz, 1/np.sqrt(6)*(px-1j*py), 0],
                                    [0, -1/np.sqrt(6)*(px+1j*py), np.sqrt(2/3)*pz, 1/np.sqrt(2)*(px-1j*py)]]) )
    
    P = gamma1_6KP/(2*m0)*(px**2 + py**2 + pz**2)
    Q = gamma2_6KP/(2*m0)*(px**2 + py**2 - 2*pz**2)
    R = np.sqrt(3)/(2*m0) * (-gamma2_6KP*(px**2 - py**2) + 2j*gamma3_6KP*((px*py)))
    S = np.sqrt(3)/(m0) * gamma3_6KP*((px-1j*py)*pz)
    H_4KP = ( -mu*np.eye(4) - np.array([[P+Q, -S, R, 0],
                                        [-np.conj(S), P-Q, 0, R],
                                        [np.conj(R), 0, P-Q, S],
                                        [0, np.conj(R), np.conj(S), P+Q]]) )
    
    H_Delta_CB = Delta_CB*np.array([[0,1],[-1,0]])
    
    H_6KP = np.block([[H_c,              H_cv4],
                      [np.conj(H_cv4.T), H_4KP]])
    
    H_BdG = np.block([[H_c,                     H_cv4,                                  H_Delta_CB,         np.zeros((2,4))],
                      [np.conj(H_cv4.T),        H_4KP,                                  np.zeros((4,2)),    np.zeros((4,4))],
                      [np.conj(H_Delta_CB.T),   np.zeros((2,4)),                        -np.conj(H_c),      -np.conj(H_cv4)],
                      [np.zeros((4,2)),         np.zeros((4,4)),                        -H_cv4.T,           -np.conj(H_4KP)]])
    
    return H_6KP, H_BdG

''' USEFUL FUNTIONS FOR PLOTTING '''

''' Functions for calculating character of the bands for each model '''
def band_composition_8KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    H_8KP, H_BdG = Hamiltonian_8KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu)
    eigvals, eigvectors = np.linalg.eigh(H_BdG)
    N = 16
    comp_HH = np.zeros(N)
    comp_LH = np.zeros(N)
    comp_SO = np.zeros(N)
    comp_EL = np.zeros(N)
    for i in range(N):
        comp_HH[i] = abs(eigvectors[2,i])**2 + abs(eigvectors[5,i])**2 + abs(eigvectors[10,i])**2 + abs(eigvectors[13,i])**2
        comp_LH[i] = abs(eigvectors[3,i])**2 + abs(eigvectors[4,i])**2 + abs(eigvectors[11,i])**2 + abs(eigvectors[12,i])**2
        comp_SO[i] = abs(eigvectors[6,i])**2 + abs(eigvectors[7,i])**2 + abs(eigvectors[14,i])**2 + abs(eigvectors[15,i])**2
        comp_EL[i] = abs(eigvectors[0,i])**2 + abs(eigvectors[1,i])**2 + abs(eigvectors[8,i])**2 + abs(eigvectors[9,i])**2

    return eigvals, comp_HH, comp_LH, comp_SO, comp_EL

def band_composition_6KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    H_6KP, H_BdG = Hamiltonian_6KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu)
    eigvals, eigvectors = np.linalg.eigh(H_BdG)
    N = 12
    comp_HH = np.zeros(N)
    comp_LH = np.zeros(N)
    comp_SO = np.zeros(N)
    for i in range(N):
        comp_HH[i] = abs(eigvectors[0,i])**2 + abs(eigvectors[3,i])**2 + abs(eigvectors[6,i])**2 + abs(eigvectors[9,i])**2
        comp_LH[i] = abs(eigvectors[1,i])**2 + abs(eigvectors[2,i])**2 + abs(eigvectors[7,i])**2 + abs(eigvectors[8,i])**2
        comp_SO[i] = abs(eigvectors[4,i])**2 + abs(eigvectors[5,i])**2 + abs(eigvectors[10,i])**2 + abs(eigvectors[11,i])**2

    return eigvals, comp_HH, comp_LH, comp_SO

def band_composition_4KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1):
    H_4KP, H_BdG = Hamiltonian_4KP_3D(kx, ky, kz, B, Delta_CB, Delta_LH, Delta_HH, mu)
    eigvals, eigvectors = np.linalg.eigh(H_BdG)
    N = 8
    comp_HH = np.zeros(N)
    comp_LH = np.zeros(N)
    for i in range(N):
        comp_HH[i] = abs(eigvectors[0,i])**2 + abs(eigvectors[3,i])**2 + abs(eigvectors[4,i])**2 + abs(eigvectors[7,i])**2
        comp_LH[i] = abs(eigvectors[1,i])**2 + abs(eigvectors[2,i])**2 + abs(eigvectors[5,i])**2 + abs(eigvectors[6,i])**2

    return eigvals, comp_HH, comp_LH

def band_composition_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu=-0.1, R0=None, disorder=[0,0,0,0]):
    H_4KP, H_BdG = Hamiltonian_4KP_2DHG(kx, ky, B, Delta_CB, Delta_LH, Delta_HH, mu, R0=R0, disorder=disorder)
    eigvals, eigvectors = np.linalg.eigh(H_BdG)
    N = 8
    comp_HH = np.zeros(N)
    comp_LH = np.zeros(N)
    for i in range(N):
        comp_HH[i] = abs(eigvectors[0,i])**2 + abs(eigvectors[3,i])**2 + abs(eigvectors[4,i])**2 + abs(eigvectors[7,i])**2
        comp_LH[i] = abs(eigvectors[1,i])**2 + abs(eigvectors[2,i])**2 + abs(eigvectors[5,i])**2 + abs(eigvectors[6,i])**2

    return eigvals, comp_HH, comp_LH

''' gap_positions: Functions to find the position of the gaps in momentum, for each model '''
''' cut_hl: Functions to find the crossing between LH and HH bands, for each model '''
def gap_positions_8KP(phi, theta, mu=-0.1):
    def aux_func_1_k(k, phi, theta, mu):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_8KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[2]

    def aux_func_2_k(k, phi, theta, mu):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_8KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[4]
    
    k_1 = fsolve(aux_func_1_k, 1e8, args=(phi, theta, mu))
    k_2 = fsolve(aux_func_2_k, 1e8, args=(phi, theta, mu))
    return k_1[0], k_2[0]

def cut_hl_8KP(phi, theta, Delta_CB, Delta_LH, Delta_HH):
    def aux_func_k(k, phi, theta, Delta_CB, Delta_LH, Delta_HH):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_8KP_3D(kx, ky, kz, [0,0,0], Delta_CB, Delta_LH, Delta_HH)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[6] - eigvals[4]
    
    k = fsolve(aux_func_k, 6e8, args=(phi, theta, Delta_CB, Delta_LH, Delta_HH))
    return k[0]

def gap_positions_6KP(phi, theta, mu=-0.1):
    def aux_func_1_k(k, phi, theta, mu):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_6KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[2]

    def aux_func_2_k(k, phi, theta, mu):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_6KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[4]
    
    k_1 = fsolve(aux_func_1_k, 1e8, args=(phi, theta, mu))
    k_2 = fsolve(aux_func_2_k, 1e8, args=(phi, theta, mu))
    return k_1[0], k_2[0]

def cut_hl_6KP(phi, theta, Delta_CB, Delta_LH, Delta_HH):
    def aux_func_k(k, phi, theta, Delta_CB, Delta_LH, Delta_HH):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_6KP_3D(kx, ky, kz, [0,0,0], Delta_CB, Delta_LH, Delta_HH)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[4] - eigvals[2]
    
    k = fsolve(aux_func_k, 7e8, args=(phi, theta, Delta_CB, Delta_LH, Delta_HH))
    return k[0]

def gap_positions_4KP(phi, theta):
    def aux_func_1_k(k, phi, theta):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_4KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[0]

    def aux_func_2_k(k, phi, theta):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_4KP_3D(kx, ky, kz, [0,0,0], 0, 0, 0)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[2]
    
    k_1 = fsolve(aux_func_1_k, 1e8, args=(phi, theta))[0]
    k_2 = fsolve(aux_func_2_k, 4e8, args=(phi, theta))[0]
    return k_1, k_2

def cut_hl_4KP(phi, theta, Delta_CB, Delta_LH, Delta_HH):
    
    def aux_func_k(k, phi, theta, Delta_CB, Delta_LH, Delta_HH):
        kx = k[0]*np.sin(theta)*np.cos(phi)
        ky = k[0]*np.sin(theta)*np.sin(phi)
        kz = k[0]*np.cos(theta)
        H, H_BdG = Hamiltonian_4KP_3D(kx, ky, kz, [0,0,0], Delta_CB, Delta_LH, Delta_HH)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[2] - eigvals[0]
    
    k = fsolve(aux_func_k, 4e8, args=(phi, theta, Delta_CB, Delta_LH, Delta_HH))[0]
    return k

def gap_positions_4KP_2DHG(phi, mu=-0.1, B=[0,0,0]):
    def aux_func_k(k, phi, i, mu):
        kx = k[0]*np.cos(phi)
        ky = k[0]*np.sin(phi)
        H, H_BdG = Hamiltonian_4KP_2DHG(kx, ky, B, 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[i]
    
    k = np.zeros(4)
    for i in range(4):
        k[i] = fsolve(aux_func_k, 1e8, args=(phi, i, mu))[0]
    return k

def gap_positions_4KP_2DHG_finite_B(phi, B, mu=-0.1):
    def aux_func_k(k, phi, B, mu):
        kx = k[0]*np.cos(phi)
        ky = k[0]*np.sin(phi)
        H, H_BdG = Hamiltonian_4KP_2DHG(kx, ky, B, 0, 0, 0, mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[4] - eigvals[3]
    
    k0 = gap_positions_4KP_2DHG(phi, mu=mu)
    
    k = np.zeros(4)
    for i in range(4):
        k[i] = fsolve(aux_func_k, k0[i], args=(phi, B, mu))[0]
    return k

def cut_hl_4KP_2DHG(phi, mu=-0.1, B=[0,0,0]):
    def aux_func_k(k, phi, mu):
        kx = k*np.cos(phi)
        ky = k*np.sin(phi)
        H, H_BdG = Hamiltonian_4KP_2DHG(kx, ky, B, 0,0,0, mu)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[3] - eigvals[2]
    
    def find_root_now(a,b):
        # tol = 1e-14*(b-a)
        c = (a + b) / 2
        it = 100000
        n = 0
        while aux_func_k(c, phi, mu) > 1e-7 and n < it:
            c = (a + b) / 2
            c1 = c - 1e-7*(b-a)
            c2 = c + 1e-7*(b-a)
            deriv = (aux_func_k(c2, phi, mu) - aux_func_k(c1, phi, mu))/(c2-c1)
            if deriv < 0:
                a = c
            else:
                b = c
            n += 1

        return c
    
    k = gap_positions_4KP_2DHG(phi, mu)
    k12 = find_root_now(k[0], k[1])
    k34 = find_root_now(k[2], k[3])
    # k12 = fsolve(aux_func_k, k[1], args=(phi, mu))[0]
    # k34 = fsolve(aux_func_k, k[3], args=(phi, mu))[0]
    return k12, k34

def cut_normal_4KP_2DHG(phi, mu=-0.1, B=[0,0,0]):
    def aux_func_k(k, phi, mu):
        kx = k[0]*np.cos(phi)
        ky = k[0]*np.sin(phi)
        H, H_BdG = Hamiltonian_4KP_2DHG(kx, ky, B, 0,0,0, mu)
        eigvals = np.linalg.eigh(H)[0]
        return eigvals[2] - eigvals[0]

    k = fsolve(aux_func_k, 5e8, args=(phi, mu))[0]
    return k

def gap_positions_2KP(phi):
    def aux_func_k(k, phi):
        kx = k[0]*np.cos(phi)
        ky = k[0]*np.sin(phi)
        H, H_BdG = Hamiltonian_2KP_2DHG(kx, ky, [0,0,0], 0, 0, 0, 0)
        eigvals = np.linalg.eigh(H_BdG)[0]
        return eigvals[2] - eigvals[1]
    
    k = fsolve(aux_func_k, 5e8, args=(phi))[0]
    return k