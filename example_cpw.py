"""
@author: Ziad (zi.hatab@gmail.com)

Example of SRM calibration using cpw synthetic data.
"""

import os
import zipfile
import copy

# via pip install
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# my codes
from srm import SRM
from multiline import multiline
from cpw import CPW

def read_waves_to_S_from_zip(zipfile_full_dir, file_name_contain):
    # read wave parameter files and convert to S-parameters (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        A = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_A' in key])
        B = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_B' in key])    
    freq = A[0].frequency
    S = rf.NetworkSet( [rf.Network(s=b.s@np.linalg.inv(a.s), frequency=freq) for a,b in zip(A,B)] )
    return S.mean_s, S.cov(), np.array([s.s for s in S])

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'        
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return [T,S[1,0]] if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return [S,T[1,1]] if pseudo else S/T[1,1]

def mobius(T, z):
    # mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t11*z + t12)/(t21*z + t22)

def mobius_inv(T, z):
    # inverse mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t12 - t22*z)/(t21*z - t11)

def Qnm(Zn, Zm):
    # Impedance transformer in T-parameters from on Eqs. (86) and (87) in
    # R. Marks and D. Williams, "A general waveguide circuit theory," 
    # Journal of Research (NIST JRES), National Institute of Standards and Technology,
    # Gaithersburg, MD, no. 97, 1992.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914227/
    Gnm = (Zm-Zn)/(Zm+Zn)
    return np.sqrt(Zn.real/Zm.real*(Zm/Zn).conjugate())/np.sqrt(1-Gnm**2)*np.array([[1, Gnm],[Gnm, 1]])

def one_port_device(f, Rdc=50, L=0, C=0, G=0, gamma=0, Zc=50, l=0, Zref=50, shunt_series=True):
    """generate a general one-port device model
    Parameters:
        f: (list or float) frequency in Hz
        Rdc: (float) DC resistance in Ohm
        L: (list or float) inductance in Henry. If list, it should match frequency points
        C: (list or float) capacitance in Farad. If list, it should match frequency points
        G: (list or float) conductance in Siemens. If list, it should match frequency points
        gamma: (list or float) propagation constant of the transmission line in Np/m.
        Zc: (list or float) characteristic impedance of the transmission line in Ohm.
        l: (float) length of the transmission line in meters.
        Zref: (list or float) reference impedance in Ohm.
        NOTE: all parameters can be list, but they should have the same length as f. Except Rdc and l. 
    """
    # make everything an array
    f = np.atleast_1d(f)
    L = np.ones_like(f)*L
    C = np.ones_like(f)*C
    G = np.ones_like(f)*G
    gamma = np.ones_like(f)*gamma
    Zc = np.ones_like(f)*Zc
    Zref = np.ones_like(f)*Zref
    
    # compute the reflection coefficient at DC (check if Rdc is infinite, i.e., open)
    rho_dc = np.ones_like(f) if np.isinf(Rdc) else (Rdc-Zref)/(Rdc+Zref)
    
    # compute the lumped elements, either series-shunt or shunt-series
    omega  = 2*np.pi*f
    Y = (G + 1j*omega*C)*Zref
    Z = 1j*omega*L/Zref
    if shunt_series:
        # see paper for equations: https://doi.org/10.1109/OJIM.2023.3315349
        # C-L configuration (for short and finite impedances)
        P = 0.5*np.array([ [ [(1-y)*(1-z)+1, (1-y)*(1+z)-1],[(1+y)*(1-z)-1, (1+y)*(1+z)+1] ] for y,z in zip(Y,Z) ])
    else:
        # L-C configuration (for open and near open impedances)
        P = 0.5*np.array([ [ [(1-y)*(1-z)+1, (y+1)*(z-1)+1], [(y-1)*(z+1)+1, (1+y)*(1+z)+1] ] for y,z in zip(Y,Z) ])

    # compute the transmission line segment as T-parameters    
    T = np.array([ Qnm(zr,zc)@np.diag([np.exp(-g*l), np.exp(g*l)])@Qnm(zc,zr) for g,zc,zr in zip(gamma, Zc, Zref)])
    
    # return the final reflection coefficient (not an skrf network)
    return np.array([mobius(t@p, rdc) for t,p,rdc in zip(T,P,rho_dc)])

def TL(l, cpw, Z01=None, Z02=None):
    """
    create an skrf network from a general transmission line model from an cpw object (see file: cpw.py)
    Parameters:
       l: (float) length of the transmission line in meters.
       cpw: (object) cpw object (see file: cpw.py)
       Z01: (list or float) reference impedance from the left side. If None, it is the same as cpw.Z0.
       Z02: (list or float) reference impedance from the right side. If None, it is the same as Z01.
    """

    f = cpw.f    # frequency points
    Z01 = cpw.Z0 if Z01 is None else np.atleast_1d(Z01)*np.ones_like(f)
    Z02 = Z01 if Z02 is None else np.atleast_1d(Z02)*np.ones_like(f)
    
    S = []
    for g,zc,z01,z02 in zip(cpw.gamma, cpw.Z0, Z01, Z02):
        T = Qnm(z01,zc)@np.diag([np.exp(-l*g), np.exp(l*g)])@Qnm(zc,z02)
        S.append(t2s(T))
    
    freq = rf.Frequency.from_f(cpw.f, unit='Hz')
    freq.unit = 'GHz'
    
    return rf.Network(s=np.array(S), frequency=freq, name=f'l={l*1e3:.2f}mm')

def embbed_error(k,X,NW):
    # embed the error box to an skrf network
    eps = np.finfo(float).eps
    new_NW = NW.copy()
    S = NW.s
    out = [s2t(s,pseudo=True) for s in S]
    T = [x[0] for x in out]
    C = [x[1] for x in out]
    S_scale = np.array([t2s( kk*XX.dot(t.flatten('F')).reshape((2,2), order='F') ) for t,kk,XX in zip(T,k,X)])
    S_new = np.array([ s*np.array([[1,1/(c+eps)],[c+eps,1]]) for s,c in zip(S_scale,C)])
    new_NW.s = S_new
    return new_NW

def skrf_from_rho(f,rho):
    freq = rf.Frequency.from_f(f, unit='hz')
    freq.unit = 'ghz'
    return rf.Network(s=rho, frequency=freq, name='')

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\FF_ISS_measurements\\'
    file_name = 'ff_ISS'
    print('Loading files... please wait!!!')
    # these data are already corrected with switch term effects
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path + f'{file_name}_thru.zip', f'{file_name}_thru')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path + f'{file_name}_line01.zip', f'{file_name}_line01')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path + f'{file_name}_line02.zip', f'{file_name}_line02')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path + f'{file_name}_line03.zip', f'{file_name}_line03')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path + f'{file_name}_line04.zip', f'{file_name}_line04')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path + f'{file_name}_line05.zip', f'{file_name}_line05')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path + f'{file_name}_open.zip', f'{file_name}_open')
    f = L1.frequency.f  # frequency axis
        
    lines = [L1,L2,L3,L4,L5,L6]
    line_lengths = [200e-6, 450e-6, 900e-6, 1800e-6, 3500e-6, 5250e-6]
    ereff_est  =  5.3
    reflect = OPEN
    reflect_est = 1
    reflect_offset = -0.1e-3
    
    cal = multiline(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est)
    cal.run_multiline()
    
    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = 9.9
    Df = 0.0002
    sig_r = 0.7
    sig_cu = 58e6 # (s/m)
    sig = sig_cu*sig_r  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    
    length = 4e-3
    line_cpw_full = TL(l=length, cpw=cpw, Z01=50)
    line_cpw_half = TL(l=length/2, cpw=cpw, Z01=50)
        
    cpw_dut = copy.copy(cpw)
    cpw_dut.w = 15e-6
    cpw_dut.update()
    DUT = TL(l=length, cpw=cpw_dut, Z01=50)
        
    length_offset = 0.2e-3
    
    L_short = 10e-12 + 1e-25*f
    C_short = 0.5e-15
    short_cpw = skrf_from_rho(f, one_port_device(f, Rdc=0, L=L_short, C=C_short, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True))
    short_cpw = rf.two_port_reflect(short_cpw, short_cpw)
    
    L_open = 0.5e-12
    C_open = 10e-15 + 1e-31*f
    open_cpw = skrf_from_rho(f, one_port_device(f, Rdc=np.inf, L=L_open, C=C_open, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=False))
    open_cpw = rf.two_port_reflect(open_cpw, open_cpw)
    
    L_match = 5e-12
    C_match = 0.5e-15
    match_cpw = skrf_from_rho(f, one_port_device(f, Rdc=50, L=L_match, C=C_match, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True))
    match_cpw = rf.two_port_reflect(match_cpw, match_cpw)
        
    SHORT = embbed_error(cal.k, cal.X, short_cpw)
    OPEN  = embbed_error(cal.k, cal.X, open_cpw)
    MATCH = embbed_error(cal.k, cal.X, match_cpw)
    
    RECIPROCAL = embbed_error(cal.k, cal.X, line_cpw_full)
    
    DUT_embbed = embbed_error(cal.k, cal.X, DUT)
    
    RECI_full_SHORT = embbed_error(cal.k, cal.X, line_cpw_full**short_cpw**line_cpw_full)
    RECI_full_OPEN  = embbed_error(cal.k, cal.X, line_cpw_full**open_cpw**line_cpw_full)
    RECI_full_MATCH = embbed_error(cal.k, cal.X, line_cpw_full**match_cpw**line_cpw_full)
    
    RECI_half_SHORT = embbed_error(cal.k, cal.X, line_cpw_half**short_cpw**line_cpw_half)
    RECI_half_OPEN  = embbed_error(cal.k, cal.X, line_cpw_half**open_cpw**line_cpw_half)
    RECI_half_MATCH = embbed_error(cal.k, cal.X, line_cpw_half**match_cpw**line_cpw_half)
    
    RECI_full_SHORT_p1 = RECI_full_SHORT.s11
    RECI_full_SHORT_p2 = RECI_full_SHORT.s22
    RECI_full_OPEN_p1  = RECI_full_OPEN.s11
    RECI_full_OPEN_p2  = RECI_full_OPEN.s22
    RECI_full_MATCH_p1 = RECI_full_MATCH.s11
    RECI_full_MATCH_p2 = RECI_full_MATCH.s22

    RECI_half_SHORT_p1 = RECI_half_SHORT.s11
    RECI_half_SHORT_p2 = RECI_half_SHORT.s22
    RECI_half_OPEN_p1  = RECI_half_OPEN.s11
    RECI_half_OPEN_p2  = RECI_half_OPEN.s22
    RECI_half_MATCH_p1 = RECI_half_MATCH.s11
    RECI_half_MATCH_p2 = RECI_half_MATCH.s22

    # SRM calibration
    # full network (at least 3 standards)
    cal = SRM(symmetric=[SHORT, OPEN, MATCH], 
              est_symmetric=[short_cpw, open_cpw, match_cpw], 
              reciprocal=RECIPROCAL,
              est_reciprocal=line_cpw_full,
              reciprocal_GammaA=[RECI_full_SHORT_p1, RECI_full_OPEN_p1, RECI_full_MATCH_p1],
              #reciprocal_GammaB=[RECI_full_SHORT_p2, RECI_full_OPEN_p2, RECI_full_MATCH_p2],
              matchA=MATCH.s11, matchB=MATCH.s22, matchA_def=match_cpw.s11, matchB_def=match_cpw.s22,
              use_half_network=False
              )
    cal.run()
    
    # half network (at least 3 standards)
    cal_half = SRM(symmetric=[SHORT, OPEN, MATCH], 
              est_symmetric=[short_cpw, open_cpw, match_cpw], 
              reciprocal=RECIPROCAL,
              est_reciprocal=line_cpw_full,
              reciprocal_GammaA=[RECI_half_SHORT_p1, RECI_half_OPEN_p1, RECI_half_MATCH_p1],
              #reciprocal_GammaB=[RECI_half_SHORT_p2, RECI_half_OPEN_p2, RECI_half_MATCH_p2],
              matchA=MATCH.s11, matchB=MATCH.s22, matchA_def=match_cpw.s11, matchB_def=match_cpw.s22,
              use_half_network=True
              )
    cal_half.run()
    
    DUT_calibrated = cal.apply_cal(DUT_embbed)
    DUT_calibrated_half = cal_half.apply_cal(DUT_embbed)
    
    err = abs(DUT.s - DUT_calibrated.s)
    err_half = abs(DUT.s - DUT_calibrated_half.s)
    
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=3, h_pad=2.5)

        ax = axs[0]
        ax.plot(f*1e-9, mag2db(err[:,0,0]), lw=2.5, marker='o', markevery=20, markersize=12, label='Full Network')
        ax.plot(f*1e-9, mag2db(err_half[:,0,0]), lw=2.5, marker='X', markevery=20, markersize=12, label='Half Network') 
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0, 150.1, 30))
        ax.set_xlim(0, 150)
        ax.set_ylabel('S11 error (dB)')
        ax.set_yticks(np.arange(-350,-249,20))
        ax.set_ylim(-350,-250)
        ax.legend(loc='lower right', ncol=1, fontsize=12)
        
        ax = axs[1]
        ax.plot(f*1e-9, mag2db(err[:,1,0]), lw=2.5, marker='o', markevery=20, markersize=12, label='Full Network')
        ax.plot(f*1e-9, mag2db(err_half[:,1,0]), lw=2.5, marker='X', markevery=20, markersize=12, label='Half Network')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel('S21 error (dB)')
        ax.set_yticks(np.arange(-350,-249,20))
        ax.set_ylim(-350,-250)
        ax.legend(loc='lower right', ncol=1, fontsize=12)
    
        #fig.savefig('error_numerical_simulation_DUT.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0)
                    
        
    plt.show()

# EOF