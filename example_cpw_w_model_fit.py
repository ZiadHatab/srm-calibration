"""
@author: Ziad (zi.hatab@gmail.com)

Example of SRM calibration using cpw synthetic data and model fit the match standard.
"""

import os
import zipfile
import copy

# via pip install
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# my code
from multiline import multiline
from srm import SRM
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
    # Gaithersburg, MD, no. 97, 1992, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914227/
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

def open_model(f,x):
    """
    A customized model to describe an open standard (cpw geometry are fixed)
    """
    # LC lumped elements
    L = x[0]*1e-12
    C = x[1]*1e-15 + x[2]*1e-30*f
    
    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = x[3]
    Df = x[4]
    sig_r = x[5]
    sig_cu = 58e6 # (s/m)
    sig = sig_cu*sig_r  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    
    length = 0.2e-3
    
    return one_port_device(f, Rdc=np.inf, L=L, C=C, l=length, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=False)
    
def short_model(f,x):
    """
    A customized model to describe a short standard (cpw geometry are fixed)
    """
    # LC lumped elements
    L = x[0]*1e-12 + x[1]*1e-24*f
    C = x[2]*1e-15
    
    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = x[3]
    Df = x[4]
    sig_r = x[5]
    sig_cu = 58e6 # (s/m)
    sig = sig_cu*sig_r  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    
    length = 0.2e-3
    
    return one_port_device(f, Rdc=0, L=L, C=C, l=length, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True)

def match_model(f,x):
    """
    A customized model to describe a match standard (cpw geometry are fixed)
    """
    # LC lumped elements
    L = x[0]*1e-12
    C = x[1]*1e-15
    
    # CPW model parameters 
    w, s, wg, t = 49.1e-6, 25.5e-6, 273.3e-6, 4.9e-6
    Dk = x[2]
    Df = x[3]
    sig_r = x[4]
    sig_cu = 58e6 # (s/m)
    sig = sig_cu*sig_r  # conductivity of Gold
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sig)
    
    length = 0.2e-3
    
    return one_port_device(f, Rdc=50, L=L, C=C, l=length, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True)

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    # load mtrl files to establish the error boxes to be used to embed the DUT SRM cal standards
    path = os.path.dirname(os.path.realpath(__file__)) + '\\FF_ISS_measurements\\'
    file_name = 'ff_ISS'
    print('Loading files... please wait!!!')
    # these data are already corrected with switch terms
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path + f'{file_name}_thru.zip', f'{file_name}_thru')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path + f'{file_name}_line01.zip', f'{file_name}_line01')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path + f'{file_name}_line02.zip', f'{file_name}_line02')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path + f'{file_name}_line03.zip', f'{file_name}_line03')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path + f'{file_name}_line04.zip', f'{file_name}_line04')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path + f'{file_name}_line05.zip', f'{file_name}_line05')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path + f'{file_name}_open.zip', f'{file_name}_open')
    f = L1.frequency.f  # frequency axis
    
    # run multiline calibration
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
    
    # create a line standard
    length = 4e-3
    line_cpw_full = TL(l=length, cpw=cpw, Z01=50)

    # create a DUT (stepped line)    
    cpw_dut = copy.copy(cpw)
    cpw_dut.w = 15e-6
    cpw_dut.update()
    DUT = TL(l=length, cpw=cpw_dut, Z01=50)
    
    # create the one-port standards
    length_offset = 0.2e-3
    
    theta_short_true = np.array([30, 10, 0.5, Dk, Df, sig_r])
    L_short = theta_short_true[0]*1e-12 + theta_short_true[1]*1e-24*f
    C_short = theta_short_true[2]*1e-15
    short_cpw = skrf_from_rho(f, one_port_device(f, Rdc=0, L=L_short, C=C_short, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True))
    short_cpw = rf.two_port_reflect(short_cpw, short_cpw)
    
    theta_open_true = np.array([1, 15, 5, Dk, Df, sig_r])
    L_open = theta_open_true[0]*1e-12
    C_open = theta_open_true[1]*1e-15 + theta_open_true[2]*1e-30*f
    open_cpw = skrf_from_rho(f, one_port_device(f, Rdc=np.inf, L=L_open, C=C_open, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=False))
    open_cpw = rf.two_port_reflect(open_cpw, open_cpw)
    
    theta_match_true = np.array([25, 1, Dk, Df, sig_r])
    L_match = theta_match_true[0]*1e-12
    C_match = theta_match_true[1]*1e-15
    match_cpw = skrf_from_rho(f, one_port_device(f, Rdc=50, L=L_match, C=C_match, l=length_offset, gamma=cpw.gamma, Zc=cpw.Z0, shunt_series=True))
    match_cpw = rf.two_port_reflect(match_cpw, match_cpw)
    
    # embed the error boxes to the DUT and standards
    SHORT = embbed_error(cal.k, cal.X, short_cpw)
    OPEN  = embbed_error(cal.k, cal.X, open_cpw)
    MATCH = embbed_error(cal.k, cal.X, match_cpw)
    
    RECIPROCAL = embbed_error(cal.k, cal.X, line_cpw_full)
    
    DUT_embbed = embbed_error(cal.k, cal.X, DUT)
    
    # only considering port-A (can be done also at port-B)
    RECI_full_SHORT_p1 = embbed_error(cal.k, cal.X, line_cpw_full**short_cpw).s11
    RECI_full_OPEN_p1  = embbed_error(cal.k, cal.X, line_cpw_full**open_cpw).s11
    RECI_full_MATCH_p1 = embbed_error(cal.k, cal.X, line_cpw_full**match_cpw).s11
    
    # you have to re-enter measurement data
    match_fit = {'meas' : [MATCH.s11.s.squeeze(), MATCH.s22.s.squeeze()],    # give this directly as S-parameters and not skrf network
                 'func' : match_model, # this is a function that takes x and f
                 'initialValues': [10, 1, 9.1, 0.00012, 0.6],
                 'bounds': [(1,50), (0.1,5), (8,11), (0.0001,0.001), (0.5, 0.9)]
                 }
    
    short_fit = match_fit.copy()
    short_fit['meas'] = [SHORT.s11.s.squeeze(), SHORT.s22.s.squeeze()]
    short_fit['func'] = short_model
    short_fit['initialValues'] = [10, 1, 1, 9.1, 0.00012, 0.6]
    short_fit['bounds'] = [(1,50), (1,20), (0.1,5), (8,11), (0.0001,0.001), (0.5, 0.9)]
    
    open_fit = match_fit.copy()
    open_fit['meas'] = [OPEN.s11.s.squeeze(), OPEN.s22.s.squeeze()]
    open_fit['func'] = open_model
    open_fit['initialValues'] = [1, 10, 1, 9.1, 0.00011, 0.61]
    open_fit['bounds'] = [(0.1,5), (1,50), (1,20), (8,11), (0.0001,0.001), (0.5, 0.9)]
    
    model_fit = [match_fit, short_fit, 
                 #open_fit
                 ]
    
    # SRM without model fit
    cal_ideal_match = SRM(symmetric=[SHORT, OPEN, MATCH], 
              est_symmetric=[short_cpw, open_cpw, match_cpw], 
              reciprocal=RECIPROCAL,
              est_reciprocal=line_cpw_full,
              reciprocal_GammaA=[RECI_full_SHORT_p1, RECI_full_OPEN_p1, RECI_full_MATCH_p1],
              matchA=MATCH.s11, matchB=MATCH.s22
              )
    cal_ideal_match.run()

    # SRM calibration with model fit
    cal_fitted_match = SRM(symmetric=[SHORT, OPEN, MATCH], 
              est_symmetric=[short_cpw, open_cpw, match_cpw], 
              reciprocal=RECIPROCAL,
              est_reciprocal=line_cpw_full,
              reciprocal_GammaA=[RECI_full_SHORT_p1, RECI_full_OPEN_p1, RECI_full_MATCH_p1],
              matchA=MATCH.s11, matchB=MATCH.s22,
              model_fit=model_fit,
              fit_max_iter=100  # this controls number of iteration taken by the DE method
              )
    cal_fitted_match.run()
    
    DUT_cal_ideal_match = cal_ideal_match.apply_cal(DUT_embbed)
    DUT_cal_fitted_match = cal_fitted_match.apply_cal(DUT_embbed)
    
    error_ideal_match = abs((DUT.s - DUT_cal_ideal_match.s)/DUT.s)
    error_fitted_match = abs((DUT.s - DUT_cal_fitted_match.s)/DUT.s)
    
    with PlotSettings(14):
        fig, axs = plt.subplots(3,2, figsize=(10,8.5))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=2.5, h_pad=2.5)
        ax = axs[0,0]
        ax.plot(f*1e-9, mag2db(DUT.s[:,0,0]), lw=2.5, marker='o', markevery=15, markersize=12, label='Actual DUT', linestyle='-')
        ax.plot(f*1e-9, mag2db(DUT_cal_ideal_match.s[:,0,0]), lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.')
        ax.plot(f*1e-9, mag2db(DUT_cal_fitted_match.s[:,0,0]), lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel('|S11| (dB)')
        ax.set_yticks(np.arange(-40,0.1,10))
        ax.set_ylim(-40,0)
        
        ax = axs[0,1]
        ax.plot(f*1e-9, mag2db(DUT.s[:,1,0]), lw=2.5, marker='o', markevery=15, markersize=12, label='Actual DUT', linestyle='-')
        ax.plot(f*1e-9, mag2db(DUT_cal_ideal_match.s[:,1,0]), lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.')
        ax.plot(f*1e-9, mag2db(DUT_cal_fitted_match.s[:,1,0]), lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel('|S21| (dB)')
        ax.set_yticks(np.arange(-4,0.1,1))
        ax.set_ylim(-4,0)
        ax.legend(loc='lower left', ncol=1, fontsize=11)
        
        ax = axs[1,0]
        ax.plot(f*1e-9, np.angle(DUT.s[:,0,0])/np.pi, lw=2.5, marker='o', markevery=15, markersize=12, label='Actual DUT', linestyle='-')
        ax.plot(f*1e-9, np.angle(DUT_cal_ideal_match.s[:,0,0])/np.pi, lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.')
        ax.plot(f*1e-9, np.angle(DUT_cal_fitted_match.s[:,0,0])/np.pi, lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel(r'arg(S11) ($\times \pi$ rad)')
        ax.set_yticks(np.arange(-1,1.1,0.4))
        ax.set_ylim(-1,1)
        
        ax = axs[1,1]
        ax.plot(f*1e-9, np.angle(DUT.s[:,1,0])/np.pi, lw=2.5, marker='o', markevery=15, markersize=12, label='Actual DUT', linestyle='-')
        ax.plot(f*1e-9, np.angle(DUT_cal_ideal_match.s[:,1,0])/np.pi, lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.')
        ax.plot(f*1e-9, np.angle(DUT_cal_fitted_match.s[:,1,0])/np.pi, lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel(r'arg(S21) ($\times \pi$ rad)')
        ax.set_yticks(np.arange(-1,1.1,0.4))
        ax.set_ylim(-1,1)
        
        ax = axs[2,0]
        ax.semilogy(f*1e-9, error_ideal_match[:,0,0], lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.', color='tab:orange')
        ax.semilogy(f*1e-9, error_fitted_match[:,0,0], lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--', color='tab:green')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel('S11 Relative Error')
        ax.set_yticks(np.logspace(-16, 2, 4))
        ax.set_ylim(1e-16,1e2)
        #ax.legend(loc='lower right', ncol=1, fontsize=12)
        
        ax = axs[2,1]
        ax.semilogy(f*1e-9, error_ideal_match[:,1,0], lw=2, marker='X', markevery=20, markersize=12, label='SRM with ideal match', linestyle='-.', color='tab:orange')
        ax.semilogy(f*1e-9, error_fitted_match[:,1,0], lw=2, marker='v', markevery=20, markersize=12, label='SRM with fitted match', linestyle='--', color='tab:green')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,150.1,30))
        ax.set_xlim(0,150)
        ax.set_ylabel('S21 Relative Error')
        ax.set_yticks(np.logspace(-16, 2, 4))
        ax.set_ylim(1e-16,1e2)
        #fig.savefig('numerical_simulation_DUT.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0)
    
    # error in the estimated parametetrs
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3.2))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=3, h_pad=2.5)
        ax = axs[0]
        err = np.array([abs( (x[1]-theta_short_true)/theta_short_true ) for x in cal_fitted_match.model_para_all_A])
        err = np.vstack(( err, abs( (cal_fitted_match.model_para_A[1]-theta_short_true)/theta_short_true ) ))
        label_text = ['L0', 'L1', 'C0', r'$\epsilon_r^\prime$', r'$\tan\delta$', r'$\sigma_r$']
        for x, tex in zip(err.T,label_text):
            ax.semilogy(x, lw=2, label=tex, linestyle='-')
        ax.set_xlabel('Optimization Index')
        ax.set_xlim(0,cal_fitted_match.fit_max_iter)
        ax.set_ylabel('Relative Error')
        ax.set_yticks(np.logspace(-16, 2, 4))
        ax.set_ylim(1e-16,1e2)
        ax.legend(loc='lower left', ncol=1, fontsize=12)
        ax.set_title('Short Standard')
        
        ax = axs[1]
        err = np.array([abs( (x[0]-theta_match_true)/theta_match_true ) for x in cal_fitted_match.model_para_all_A])
        err = np.vstack(( err, abs( (cal_fitted_match.model_para_A[0]-theta_match_true)/theta_match_true ) ))
        label_text = ['L0', 'C0', r'$\epsilon_r^\prime$', r'$\tan\delta$', r'$\sigma_r$']
        for x, tex in zip(err.T,label_text):
            ax.semilogy(x, lw=2, label=tex, linestyle='-')
        ax.set_xlabel('Optimization Index')
        ax.set_xlim(0,cal_fitted_match.fit_max_iter)
        ax.set_ylabel('Relative Error')
        ax.set_yticks(np.logspace(-16, 2, 4))
        ax.set_ylim(1e-16,1e2)
        ax.legend(loc='lower left', ncol=1, fontsize=12)
        ax.set_title('Match Standard') 
        #fig.savefig('error_in_parameters.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0)
        
        """
        This for when using all three standards in optimization
        with PlotSettings(14):
            # Create figure with set size
            fig = plt.figure(figsize=(10,6), dpi=600)
            
            # Create 2x2 grid, with spans for top row subplots
            gs = fig.add_gridspec(2, 4)
            
            # Top left plot - Short Standard
            ax1 = fig.add_subplot(gs[0,:2])
            err = np.array([abs( (x[1]-theta_short_true)/theta_short_true ) for x in cal_fitted_match.model_para_all_A])
            err = np.vstack(( err, abs( (cal_fitted_match.model_para_A[1]-theta_short_true)/theta_short_true ) ))
            label_text = ['L0', 'L1', 'C0', r'$\epsilon_r^\prime$', r'$\tan\delta$', r'$\sigma_r$']
            for x, tex in zip(err.T,label_text):
                ax1.semilogy(x, lw=2, label=tex, linestyle='-')
            ax1.set_xlabel('Optimization Index')
            ax1.set_xlim(0,cal_fitted_match.fit_max_iter)
            ax1.set_ylabel('Relative Error')
            ax1.set_yticks(np.logspace(-16, 2, 4))
            ax1.set_ylim(1e-16,1e2)
            ax1.legend(loc='lower left', ncol=2, fontsize=12)
            ax1.set_title('Short Standard')
            
            # Top right plot - Match Standard  
            ax2 = fig.add_subplot(gs[0,2:])
            err = np.array([abs( (x[0]-theta_match_true)/theta_match_true ) for x in cal_fitted_match.model_para_all_A])
            err = np.vstack(( err, abs( (cal_fitted_match.model_para_A[0]-theta_match_true)/theta_match_true ) ))
            label_text = ['L0', 'C0', r'$\epsilon_r^\prime$', r'$\tan\delta$', r'$\sigma_r$']
            for x, tex in zip(err.T,label_text):
                ax2.semilogy(x, lw=2, label=tex, linestyle='-')
            ax2.set_xlabel('Optimization Index')
            ax2.set_xlim(0,cal_fitted_match.fit_max_iter)
            ax2.set_ylabel('Relative Error')
            ax2.set_yticks(np.logspace(-16, 2, 4))
            ax2.set_ylim(1e-16,1e2)
            ax2.legend(loc='lower left', ncol=2, fontsize=12)
            ax2.set_title('Match Standard')
            
            # Bottom center plot spanning both columns
            ax3 = fig.add_subplot(gs[1,1:3])
            err = np.array([abs( (x[2]-theta_open_true)/theta_open_true ) for x in cal_fitted_match.model_para_all_A])
            err = np.vstack(( err, abs( (cal_fitted_match.model_para_A[2]-theta_open_true)/theta_open_true ) ))
            label_text = ['L0', 'C0', 'C1', r'$\epsilon_r^\prime$', r'$\tan\delta$', r'$\sigma_r$']
            for x, tex in zip(err.T,label_text):
                ax3.semilogy(x, lw=2, label=tex, linestyle='-')
            ax3.set_xlabel('Optimization Index')
            ax3.set_xlim(0,cal_fitted_match.fit_max_iter)
            ax3.set_ylabel('Relative Error')
            ax3.set_yticks(np.logspace(-16, 2, 4))
            ax3.set_ylim(1e-16,1e2)
            ax3.legend(loc='lower left', ncol=2, fontsize=12)
            ax3.set_title('Open Standard')

            fig.tight_layout()
        """
        
    plt.show()
    # EOF