"""
@author: Ziad (zi.hatab@gmail.com)

Example of SRM calibration using microstrip measurements and automatic model fit of the match standard.
"""

import os

# via pip install
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

# my code
from srm import SRM

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

def mobius(T, z):
    # mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t11*z + t12)/(t21*z + t22)

def mobius_inv(T, z):
    # inverse mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t12 - t22*z)/(t21*z - t11)

def shunt_series(y,z, shunt_series=True):
    # T-parameters of a shunt-series network (y and z are normalized conductance and impedance)
    if shunt_series:
        # see paper for equations: https://doi.org/10.1109/OJIM.2023.3315349
        # C-L configuration (for short and finite impedances)
        P = 0.5*np.array([ [(1-y)*(1-z)+1, (1-y)*(1+z)-1],[(1+y)*(1-z)-1, (1+y)*(1+z)+1] ])
    else:
        # L-C configuration (for open and near open impedances)
        P = 0.5*np.array([ [(1-y)*(1-z)+1, (y+1)*(z-1)+1], [(y-1)*(z+1)+1, (1+y)*(1+z)+1] ])
    return P

def open_model(f,x):
    """
    A customized model to describe an open standard
    """
    # LC lumped elements
    L = x[0]*1e-12
    C = x[1]*1e-15

    Zref = 50

    omega  = 2*np.pi*f
    Y = 1j*omega*C*Zref
    Z = 1j*omega*L/Zref

    Gamma = []
    for inx, _ in enumerate(f):
        P1 = shunt_series(Y[inx], Z[inx], shunt_series=False)
        Gamma.append(mobius(P1, 1))
    
    return np.array(Gamma)

def short_model(f,x):
    """
    A customized model to describe a short standard
    """
    # LC lumped elements
    L = x[0]*1e-12
    C = x[1]*1e-15

    Zref = 50

    omega  = 2*np.pi*f
    Y = 1j*omega*C*Zref
    Z = 1j*omega*L/Zref

    Gamma = []
    for inx, _ in enumerate(f):
        P1 = shunt_series(Y[inx], Z[inx], shunt_series=True)
        Gamma.append(mobius(P1, -1))
    
    return np.array(Gamma)

def match_model(f,x):
    """
    A customized model to describe a match standard
    """
    Zref = 50
    
    # LC lumped elements
    L1 = x[0]*1e-12
    C1 = x[1]*1e-15
    L2 = x[2]*1e-12
    C2 = x[3]*1e-15

    omega  = 2*np.pi*f
    Y1 = 1j*omega*C1*Zref
    Z1 = 1j*omega*L1/Zref
    Y2 = 1j*omega*C2*Zref
    Z2 = 1j*omega*L2/Zref
    
    Rdc = 49
    
    Ldc = x[4]*1e-12
    Cdc = x[5]*1e-15
    
    Lvia = x[6]*1e-12
    
    Zdc = (Rdc + 1j*omega*Ldc)/(1 + (Rdc + 1j*omega*Ldc)*(1j*omega*Cdc))
    Zdc = Zdc/Zref
    Zvia = 1j*omega*Lvia
    rho_via = (Zvia - Zref)/(Zvia + Zref)
    Gamma = []
    for inx, _ in enumerate(f):
        P1 = shunt_series(Y1[inx], Z1[inx], shunt_series=True)
        Pdc = shunt_series(0, Zdc[inx], shunt_series=True)
        P2 = shunt_series(Y2[inx], Z2[inx], shunt_series=False)
        Gamma.append(mobius(P1@Pdc@P2, rho_via[inx]))
    
    return np.array(Gamma)

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    p = np.array([[0,1],[1,0]])
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\Microstrip_measurements\\'

    print('Loading files... please wait!!!')
    # loading multiline trl lines
    L1 = rf.Network(path + 'trl_line_0_0mm.s2p')
    L2 = rf.Network(path + 'trl_line_0_5mm.s2p')
    L3 = rf.Network(path + 'trl_line_4_0mm.s2p')
    L4 = rf.Network(path + 'trl_line_5_5mm.s2p')
    L5 = rf.Network(path + 'trl_line_6_5mm.s2p')
    L6 = rf.Network(path + 'trl_line_8_5mm.s2p')

    # loading reflect stand
    open_0mm = rf.Network(path + 'trl_open_0_0mm.s2p')
    open_0mm = rf.Network(path + 'srm_open.s2p')
    freq = L1.frequency
    f = freq.f  # frequency axis
        
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = np.array([0, 0.5, 4, 5.5, 6.5, 8.5])*1e-3
    reflect = open_0mm
    reflect_offset = 0
    reflect_est = 1
    
    # run multiline cal
    cal_trl = rf.TUGMultilineTRL(line_meas=lines, line_lengths=line_lengths, er_est=2.5, 
                             reflect_meas=reflect, reflect_est=reflect_est, reflect_offset=reflect_offset)
    
    # load microstrip sim data (for impedance renormalize)
    csv_data = np.loadtxt(path + 'microstrip_sim_gamma_z0.csv', delimiter=',', skiprows=1)
    f_sim = csv_data[:,0]  # Hz
    gamma_sim = csv_data[:,1] + 1j*csv_data[:,2] 
    Z0_sim    = csv_data[:,3] + 1j*csv_data[:,4] 
    # interpolate simulation data to match measurement frequencies
    gamma_sim_interp = np.interp(f, f_sim, gamma_sim.real) + 1j*np.interp(f, f_sim, gamma_sim.imag)
    Z0_sim_interp    = np.interp(f, f_sim, Z0_sim.real) + 1j*np.interp(f, f_sim, Z0_sim.imag)
    
    RL_sim = gamma_sim_interp*Z0_sim_interp
    Z0_est = RL_sim/cal_trl.gamma
    cal_trl.renormalize(z0_old=Z0_est, z0_new=50*np.ones_like(f))
    
    # load SRM standards
    # symmetric standards
    srm_line = rf.Network(path + 'srm_line.s2p')
    srm_open = rf.Network(path + 'srm_open.s2p')
    srm_short = rf.Network(path + 'srm_short.s2p')
    srm_match = rf.Network(path + 'srm_match.s2p')
    # network-load standards
    srm_offset_open_portA = rf.Network(path + 'srm_offset_open_portA.s2p').s11
    srm_offset_short_portA = rf.Network(path + 'srm_offset_short_portA.s2p').s11
    srm_offset_match_portA = rf.Network(path + 'srm_offset_match_portA.s2p').s11
    
    # estimate values (in practice you would provide numerical estimate, e.g., using skrf Media)
    short_est = cal_trl.apply_cal(srm_short).s11
    open_est  = cal_trl.apply_cal(srm_open).s11
    match_est = cal_trl.apply_cal(srm_match)  
    line_est  = cal_trl.apply_cal(srm_line)

    # DUT (stepped impedance line)
    dut = rf.Network(path + 'dut_stepline.s2p')
    
    # you have to re-enter measurement data
    match_fit = {'meas' : [srm_match.s[:,0,0], srm_match.s[:,1,1]],    # give this directly as S-parameters and not skrf network
                 'func' : match_model, # this is a function that takes x and f
                 'initialValues': [1,1,1,1,10,10,0],
                 'bounds': [(1,1e2), (1,1e2), (1,1e2), (1,1e2), (1e1,5e2), (1e1,5e2), (0,1e1)]
                 }
    
    short_fit = match_fit.copy()
    short_fit['meas'] = [srm_short.s[:,0,0], srm_short.s[:,1,1]]
    short_fit['func'] = short_model
    short_fit['initialValues'] = [1, 1]
    short_fit['bounds'] = [(0,1e2), (0,1e3)]
    
    open_fit = match_fit.copy()
    open_fit['meas'] = [srm_open.s[:,0,0], srm_open.s[:,1,1]]
    open_fit['func'] = open_model
    open_fit['initialValues'] = [1, 1]
    open_fit['bounds'] = [(0,1e2), (0,1e2)]
    
    model_fit = [match_fit, 
                 short_fit, 
                 open_fit
                 ]

    # SRM calibration
    cal_ideal_match = SRM(symmetric=[srm_short, srm_open, srm_match], 
              est_symmetric=[short_est, open_est], 
              reciprocal=srm_line,
              est_reciprocal=line_est,
              reciprocal_GammaA=[srm_offset_short_portA, srm_offset_open_portA, srm_offset_match_portA],
              matchA=srm_match.s11, matchB=srm_match.s22,
              )
    cal_ideal_match.run()
    
    cal_fitted_match = SRM(symmetric=[srm_short, srm_open, srm_match], 
              est_symmetric=[short_est, open_est], 
              reciprocal=srm_line,
              est_reciprocal=line_est,
              reciprocal_GammaA=[srm_offset_short_portA, srm_offset_open_portA, srm_offset_match_portA],
              matchA=srm_match.s11, matchB=srm_match.s22, 
              model_fit=model_fit,
              model_fit_split=False
              )
    cal_fitted_match.run()

    cal_trl_match = SRM(symmetric=[srm_short, srm_open, srm_match], 
              est_symmetric=[short_est, open_est], 
              reciprocal=srm_line,
              est_reciprocal=line_est,
              reciprocal_GammaA=[srm_offset_short_portA, srm_offset_open_portA, srm_offset_match_portA],
              matchA=srm_match.s11, matchB=srm_match.s22, 
              matchA_def=match_est.s11, matchB_def=match_est.s22,
              )
    cal_trl_match.run()

    dut_cal_trl = cal_trl.apply_cal(dut)
    dut_cal_ideal_match  = cal_ideal_match.apply_cal(dut) 
    dut_cal_fitted_match = cal_fitted_match.apply_cal(dut)
    dut_cal_trl_match    = cal_trl_match.apply_cal(dut)
    with PlotSettings(14):
        fig, axs = plt.subplots(3,2, figsize=(10,8.5))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=2.5, h_pad=2.5)
        ax = axs[0,0]
        ax.plot(f*1e-9, mag2db(dut_cal_trl.s[:,0,0]), lw=2, marker='o', markevery=20, markersize=12, label='TRL (reference)', linestyle='-')
        ax.plot(f*1e-9, mag2db(dut_cal_ideal_match.s[:,0,0]), lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.')
        ax.plot(f*1e-9, mag2db(dut_cal_fitted_match.s[:,0,0]), lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--')
        ax.plot(f*1e-9, mag2db(dut_cal_trl_match.s[:,0,0]), lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel('|S11| (dB)')
        ax.set_yticks(np.arange(-45,5.1,10))
        ax.set_ylim(-45,5)
        
        ax = axs[0,1]
        ax.plot(f*1e-9, mag2db(dut_cal_trl.s[:,1,0]), lw=2, marker='o', markevery=20, markersize=12, label='TRL (reference)', linestyle='-')
        ax.plot(f*1e-9, mag2db(dut_cal_ideal_match.s[:,1,0]), lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.')
        ax.plot(f*1e-9, mag2db(dut_cal_fitted_match.s[:,1,0]), lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--')
        ax.plot(f*1e-9, mag2db(dut_cal_trl_match.s[:,1,0]), lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel('|S21| (dB)')
        ax.set_yticks(np.arange(-10,5.1,3))
        ax.set_ylim(-10,5)
        #ax.legend(loc='lower left', ncol=1)
        
        ax = axs[1,0]
        ax.plot(f*1e-9, np.angle(dut_cal_trl.s[:,0,0])/np.pi, lw=2, marker='o', markevery=20, markersize=12, label='TRL (reference)', linestyle='-')
        ax.plot(f*1e-9, np.angle(dut_cal_ideal_match.s[:,0,0])/np.pi, lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.')
        ax.plot(f*1e-9, np.angle(dut_cal_fitted_match.s[:,0,0])/np.pi, lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--')
        ax.plot(f*1e-9, np.angle(dut_cal_trl_match.s[:,0,0])/np.pi, lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel(r'arg(S11) ($\times \pi$ rad)')
        ax.set_yticks(np.arange(-1,1.1,0.4))
        ax.set_ylim(-1,1)
        
        ax = axs[1,1]
        ax.plot(f*1e-9, np.angle(dut_cal_trl.s[:,1,0])/np.pi, lw=2, marker='o', markevery=20, markersize=12, label='TRL (reference)', linestyle='-')
        ax.plot(f*1e-9, np.angle(dut_cal_ideal_match.s[:,1,0])/np.pi, lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.')
        ax.plot(f*1e-9, np.angle(dut_cal_fitted_match.s[:,1,0])/np.pi, lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--')
        ax.plot(f*1e-9, np.angle(dut_cal_trl_match.s[:,1,0])/np.pi, lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel(r'arg(S21) ($\times \pi$ rad)')
        ax.set_yticks(np.arange(-1,1.1,0.4))
        ax.set_ylim(-1,1)
        
        ax = axs[2,0]
        error_trl = mag2db(dut_cal_trl.s - dut_cal_trl.s)
        error_ideal = mag2db(dut_cal_ideal_match.s - dut_cal_trl.s)
        error_fitted = mag2db(dut_cal_fitted_match.s - dut_cal_trl.s)
        error_trl_match = mag2db(dut_cal_trl_match.s - dut_cal_trl.s)
        
        ax.plot(f*1e-9, error_ideal[:,0,0], lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.', color='tab:orange')
        ax.plot(f*1e-9, error_fitted[:,0,0], lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--', color='tab:green')
        ax.plot(f*1e-9, error_trl_match[:,0,0], lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--', color='tab:red')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel('S11 error (dB)')
        #ax.set_yticks(np.logspace(-4, 2, 4))
        ax.set_ylim(-50,0)
        
        ax = axs[2,1]
        ax.plot(f*1e-9, error_ideal[:,1,0], lw=2, marker='X', markevery=20, markersize=12, label='SRM ideal match', linestyle='-.', color='tab:orange')
        ax.plot(f*1e-9, error_fitted[:,1,0], lw=2, marker='v', markevery=20, markersize=12, label='SRM fitted match', linestyle='--', color='tab:green')
        ax.plot(f*1e-9, error_trl_match[:,1,0], lw=2, marker='^', markevery=20, markersize=12, label='SRM TRL match', linestyle='--', color='tab:red')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel('S21 error (dB)')
        #ax.set_yticks(np.logspace(-4, 2, 4))
        ax.set_ylim(-50,0)
        # Add single legend from first subplot only
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, .98), loc='lower center', ncol=4)
        #fig.savefig('dut_srm_trl.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0)

    # error in the estimated parametetrs
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3.2))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=3, h_pad=2.5)
        ax = axs[0]
        ax.plot(f*1e-9, mag2db(match_est.s[:,0,0]), lw=2, marker='o', markevery=20, markersize=12, label='Port-A match from TRL', linestyle='-')
        ax.plot(f*1e-9, mag2db(match_est.s[:,1,1]), lw=2, marker='X', markevery=20, markersize=12, label='Port-B match from TRL', linestyle='-')
        ax.plot(f*1e-9, mag2db(cal_fitted_match.model_eval_A[0]), lw=2, marker='v', markevery=20, markersize=12, label='Port-A SRM model fit', linestyle='--')        
        ax.plot(f*1e-9, mag2db(cal_fitted_match.model_eval_B[0]), lw=2, marker='^', markevery=20, markersize=12, label='Port-B SRM model fit', linestyle='--')    
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel('|S11| (dB)')
        ax.set_yticks(np.arange(-40,0.1,10))
        ax.set_ylim(-40,0)
        ax.legend(loc='lower right', ncol=1)
        
        ax = axs[1]
        ax.plot(f*1e-9, np.angle(match_est.s[:,0,0])/np.pi, lw=2, marker='o', markevery=20, markersize=12, label='Port-A match from TRL', linestyle='-')
        ax.plot(f*1e-9, np.angle(match_est.s[:,1,1])/np.pi, lw=2, marker='X', markevery=20, markersize=12, label='Port-B match from TRL', linestyle='-')
        ax.plot(f*1e-9, np.angle(cal_fitted_match.model_eval_A[0])/np.pi, lw=2, marker='v', markevery=20, markersize=12, label='Port-A SRM model fit', linestyle='--')
        ax.plot(f*1e-9, np.angle(cal_fitted_match.model_eval_B[0])/np.pi, lw=2, marker='^', markevery=20, markersize=12, label='Port-B SRM model fit', linestyle='--')        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,50.1,10))
        ax.set_xlim(0,50)
        ax.set_ylabel(r'arg(S11) ($\times \pi$ rad)')
        ax.set_yticks(np.arange(-1,0.1,0.2))
        ax.set_ylim(-1,0)
        #fig.savefig('match_fit_trl.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0)

    plt.show()
# EOF