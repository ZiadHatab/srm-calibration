"""
@author: Ziad (zi.hatab@gmail.com)

Example of SRM calibration using coaxial measurements.
"""

import os
import zipfile

# via pip install
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import metas_unclib as munc
munc.use_linprop()  # this is to load the uncertainties of verification kits.

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

def read_s2p_to_S_from_zip(zipfile_full_dir, file_name_contain, use_switch_terms=True):
    # read s2p files and convert to S-parameters with switch terms corrected (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        S_param = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_S_param' in key])
        switch = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_switch' in key])

    freq = S_param[0].frequency
    
    if use_switch_terms:
        S = rf.NetworkSet( [rf.Network(s=np.array([np.dot([[x[0,0],x[0,1]],
                                                           [x[1,0],x[1,1]]], np.linalg.inv([[1, x[0,1]*y[0,1]], [x[1,0]*y[1,0], 1]]))
                                                           for x,y in zip(S.s, sw.s)]), frequency=freq) 
                                                                              for S,sw in zip(S_param,switch)] )
    else:
        S = rf.NetworkSet( [rf.Network(s=np.array([[[x[0,0],x[0,1]],
                                                    [x[1,0],x[1,1]]] for x in S.s]), frequency=freq) for S in S_param ] )
    
    return S.mean_s, S.cov(), np.array([s.s for s in S])

def make_metas_from_csv(file_path):
    df  = pd.read_csv(file_path)
    f   = df.values.T[0]
    S   = df.values.T[1] + 1j*df.values.T[2]
    Cov = np.array([ [[x[0],x[2]],[x[1],x[3]]] for x in df.values[:,3:]])
    S_metas = np.array([munc.ucomplex(s,covariance=cov) for s,cov in zip(S,Cov)])
    return S_metas, f

def metas_group_delay(metas, f):
    omega = 2*np.pi*f
    return -munc.umath.imag( np.diff(metas,prepend=2*metas[0]-metas[1])/np.diff(omega, prepend=2*omega[0]-omega[1])/metas )

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    p = np.array([[0,1],[1,0]])
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\Coaxial_40ghz_measurements\\'
    fmax = 40
    fmin = 0.1
    
    print('Loading files... please wait!!!')
    open_p1 = read_s2p_to_S_from_zip(path + 'open_p1.zip', 'open_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    open_p2 = read_s2p_to_S_from_zip(path + 'open_p2.zip', 'open_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    OPEN = rf.two_port_reflect(open_p1, open_p2)
    
    short_p1 = read_s2p_to_S_from_zip(path + 'short_p1.zip', 'short_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    short_p2 = read_s2p_to_S_from_zip(path + 'short_p2.zip', 'short_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    SHORT = rf.two_port_reflect(short_p1, short_p2)
    
    match_p1 = read_s2p_to_S_from_zip(path + 'match_p1.zip', 'match_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    match_p2 = read_s2p_to_S_from_zip(path + 'match_p2.zip', 'match_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    MATCH = rf.two_port_reflect(match_p1, match_p2)
    
    dut_mismatch_p1 = read_s2p_to_S_from_zip(path + 'mismatch_p1.zip', 'mismatch_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    dut_mismatch_p2 = read_s2p_to_S_from_zip(path + 'mismatch_p2.zip', 'mismatch_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    dut_mismatch = rf.two_port_reflect(dut_mismatch_p1, dut_mismatch_p2)
    
    dut_offsetshort_p1 = read_s2p_to_S_from_zip(path + 'offsetshort_p1.zip', 'offsetshort_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    dut_offsetshort_p2 = read_s2p_to_S_from_zip(path + 'offsetshort_p2.zip', 'offsetshort_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    dut_offsetshort = rf.two_port_reflect(dut_offsetshort_p1, dut_offsetshort_p2)
    
    line_open_p1 = read_s2p_to_S_from_zip(path + 'thru_open_p1.zip', 'thru_open_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    line_open_p2 = read_s2p_to_S_from_zip(path + 'thru_open_p2.zip', 'thru_open_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    
    line_short_p1 = read_s2p_to_S_from_zip(path + 'thru_short_p1.zip', 'thru_short_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    line_short_p2 = read_s2p_to_S_from_zip(path + 'thru_short_p2.zip', 'thru_short_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    
    line_match_p1 = read_s2p_to_S_from_zip(path + 'thru_match_p1.zip', 'thru_match_p1')[0].s11[f'{fmin}ghz-{fmax}ghz']
    line_match_p2 = read_s2p_to_S_from_zip(path + 'thru_match_p2.zip', 'thru_match_p2')[0].s22[f'{fmin}ghz-{fmax}ghz']
    
    line = read_s2p_to_S_from_zip(path + 'thru.zip', 'thru')[0][f'{fmin}ghz-{fmax}ghz']
    
    freq = line.frequency
    f = freq.f  # frequency axis
    
    # ideal data of the standards
    open_ideal  = rf.Network(path + 'open(f)_101165.s1p').s11[f'{fmin}ghz-{fmax}ghz']
    short_ideal = rf.Network(path + 'short(f)_101180.s1p').s11[f'{fmin}ghz-{fmax}ghz']
    load_ideal  = rf.Network(path + 'match(f)_101170.s1p').s11[f'{fmin}ghz-{fmax}ghz']
    line_ideal  = rf.Network(path + 'Thru(ff)_101504.s2p')[f'{fmin}ghz-{fmax}ghz']
    
    mismatch_ideal = rf.Network(path + 'MISMATCH_FEMALE_ZVZ429_1319.1360.00_101170.s1p')[f'{fmin}ghz-{fmax}ghz'].interpolate(freq, coords='polar')
    mismatch_ideal = rf.two_port_reflect(mismatch_ideal)
    offsetshort_ideal = rf.Network(path + 'OFFSET_SHORT_FEMALE_ZVZ429_1319.1347.00_101183.s1p')[f'{fmin}ghz-{fmax}ghz'].interpolate(freq, coords='polar')
    offsetshort_ideal = rf.two_port_reflect(offsetshort_ideal)
    
    S_mismacth_metas, f_mismacth = make_metas_from_csv(path + 'mismatch_female.csv')
    S_offsetshort_metas, f_offsetshort = make_metas_from_csv(path + 'offsetshort_female.csv')
    
    # SRM calibration
    cal_srm = SRM(symmetric=[SHORT, OPEN, MATCH], 
              est_symmetric=[short_ideal, open_ideal, load_ideal], 
              reciprocal=line,
              est_reciprocal=line_ideal,
              reciprocal_GammaA=[line_short_p1, line_open_p1, line_match_p1], 
              #reciprocal_GammaB=[line_short_p2, line_open_p2, line_match_p2],
              matchA=match_p1, matchB=match_p2, 
              matchA_def=load_ideal, matchB_def=load_ideal, 
              )
    cal_srm.run()
    
    # SOLR calibration
    measured = [SHORT, OPEN, MATCH, line]
    ideals   = [rf.two_port_reflect(short_ideal), rf.two_port_reflect(open_ideal), rf.two_port_reflect(load_ideal), line_ideal]
    
    cal_solr = rf.UnknownThru(measured, ideals)
    cal_solr.run()
    
    cal_srm.error_coef()
    SRM  = cal_srm.coefs
    SOLR = cal_solr.coefs
    
    # difference in the error terms     
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,3))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=3, h_pad=2.5)
        
        ax = axs[0]
        err = mag2db(SOLR['forward directivity'] - SRM['EDF'])
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='Forward directivity')
        err = mag2db(SOLR['forward source match'] - SRM['ESF'])
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='Forward source match')
        err = mag2db(SOLR['forward reflection tracking'] - SRM['ERF'])
        ax.plot(f*1e-9, err, lw=2.5, marker='d', markevery=30, markersize=12,
                label='Forward reflection tracking')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('Error (dB)')
        ax.set_yticks(np.arange(-150,0.1,30))
        ax.set_ylim(-150,0)
        ax.legend(loc='lower right', ncol=1, fontsize=12)
        
        ax = axs[1]
        err = mag2db(SOLR['reverse directivity'] - SRM['EDR'])
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='Reverse directivity')
        err = mag2db(SOLR['reverse source match'] - SRM['ESR'])
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='Reverse source match')
        err = mag2db(SOLR['reverse reflection tracking'] - SRM['ERR'])
        ax.plot(f*1e-9, err, lw=2.5, marker='d', markevery=30, markersize=12,
                label='Reverse reflection tracking')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('Error (dB)')
        ax.set_yticks(np.arange(-150,0.1,30))
        ax.set_ylim(-150,0)
        ax.legend(loc='lower right', ncol=1, fontsize=12)
    
        #fig.savefig('error_calibration_comparison.pdf', format='pdf', dpi=300, 
        #        bbox_inches='tight', pad_inches = 0)
    
    # calibrated line (adapter) standard
    line_cal_solr = cal_solr.apply_cal(line)
    line_cal_srm  = cal_srm.apply_cal(line)
    with PlotSettings(14):
        fig, axs = plt.subplots(3,2, figsize=(10,8.5))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=2, h_pad=2.5)
        ax = axs[0,0]
        val = mag2db(line_ideal.s[:,0,0])
        ax.plot(f*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        
        val = mag2db(line_cal_solr.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = mag2db(line_cal_srm.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('|S11| (dB)')
        ax.set_yticks(np.arange(-80,0.1,20))
        ax.set_ylim(-80,0)
        ax.legend(loc='upper left', ncol=1, fontsize=12)
        
        ax = axs[0,1]
        val = mag2db(line_ideal.s[:,1,0])
        ax.plot(f*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        val = mag2db(line_cal_solr.s[:,1,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = mag2db(line_cal_srm.s[:,1,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('|S21| (dB)')
        ax.set_yticks(np.arange(-0.3,0.11,.1))
        ax.set_ylim(-0.3,0.1)
        
        ax = axs[1,0]
        val = line_ideal.group_delay.real[:,0,0]*1e9
        ax.plot(f*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        val = line_cal_solr.group_delay.real[:,0,0]*1e9
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = line_cal_srm.group_delay.real[:,0,0]*1e9
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('gd(S11) (ns)')
        ax.set_ylim(-4,4)
        ax.set_yticks(np.arange(-4,4.1,2))
        
        ax = axs[1,1]
        val = line_ideal.group_delay.real[:,1,0]*1e12
        ax.plot(f*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        val = line_cal_solr.group_delay.real[:,1,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = line_cal_srm.group_delay.real[:,1,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('gd(S21) (ps)')
        ax.set_ylim(70,85)
        ax.set_yticks(np.arange(70,86,3))
        
        ax = axs[2,0]
        err = mag2db((line_ideal.s- line_cal_solr.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        err = mag2db((line_ideal.s- line_cal_srm.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('S11 error (dB)')
        ax.set_yticks(np.arange(-80,0.1,20))
        ax.set_ylim(-80,0)
        
        ax = axs[2,1]
        err = mag2db((line_ideal.s- line_cal_solr.s))[:,1,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        err = mag2db((line_ideal.s- line_cal_srm.s))[:,1,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('S21 error (dB)')
        ax.set_yticks(np.arange(-80,0.1,20))
        ax.set_ylim(-80,0)
        
        #fig.savefig('solr_vs_srm_calibrated_line.pdf', format='pdf', dpi=300, 
        #        bbox_inches='tight', pad_inches = 0)
    
    # calibrated verification kits
    mismatch_cal_solr = cal_solr.apply_cal(dut_mismatch).s11
    mismatch_cal_srm  = cal_srm.apply_cal(dut_mismatch).s11
    offsetshort_cal_solr = cal_solr.apply_cal(dut_offsetshort).s11
    offsetshort_cal_srm  = cal_srm.apply_cal(dut_offsetshort).s11
    k = 2
    with PlotSettings(14):
        fig, axs = plt.subplots(3,2, figsize=(10,8.5))        
        fig.set_dpi(600)
        fig.tight_layout(w_pad=2, h_pad=2.5)
        
        ax = axs[0,0]
        metas = mag2db(S_mismacth_metas)
        val = munc.get_value(metas)
        std = munc.get_stdunc(metas)
        ax.plot(f_mismacth*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        ax.fill_between(f_mismacth*1e-9, val-k*std, val+k*std, alpha=0.3, color='black')
        val = mag2db(mismatch_cal_solr.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = mag2db(mismatch_cal_srm.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('|S11| (dB)')
        ax.set_yticks(np.arange(-30, -14, 5))
        ax.set_ylim(-30,-15)
        ax.legend(loc='lower left', ncol=1, fontsize=12)
        ax.set_title('Mismatch')
        
        ax = axs[0,1]
        metas = mag2db(S_offsetshort_metas)
        val = munc.get_value(metas)
        std = munc.get_stdunc(metas)
        ax.plot(f_mismacth*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        ax.fill_between(f_mismacth*1e-9, val-k*std, val+k*std, alpha=0.3, color='black')
        val = mag2db(offsetshort_cal_solr.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = mag2db(offsetshort_cal_srm.s[:,0,0])
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('|S11| (dB)')
        ax.set_ylim(-.6,.6)
        ax.set_yticks(np.arange(-0.6, 0.61, 0.3))
        ax.set_title('Offset Short')
                
        ax = axs[1,0]
        metas = metas_group_delay(S_mismacth_metas, f_mismacth)*1e12
        val = munc.get_value(metas)
        std = munc.get_stdunc(metas)
        ax.plot(f_mismacth*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        ax.fill_between(f_mismacth*1e-9, val-k*std, val+k*std, alpha=0.3, color='black')
        val = mismatch_cal_solr.group_delay.real[:,0,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = mismatch_cal_srm.group_delay.real[:,0,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('gd(S11) (ps)')
        ax.set_ylim(-200,300)
        ax.set_yticks(np.arange(-200,301,100))
        
        ax = axs[1,1]
        metas = metas_group_delay(S_offsetshort_metas, f_offsetshort)*1e12
        val = munc.get_value(metas)
        std = munc.get_stdunc(metas)
        ax.plot(f_offsetshort*1e-9, val, lw=3, label='Reference', linestyle='-', color='black')
        ax.fill_between(f_offsetshort*1e-9, val-k*std, val+k*std, alpha=0.3, color='black')
        val = offsetshort_cal_solr.group_delay.real[:,0,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        val = offsetshort_cal_srm.group_delay.real[:,0,0]*1e12
        ax.plot(f*1e-9, val, lw=2.5, marker='X', markevery=30, markersize=10,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('gd(S11) (ps)')
        ax.set_ylim(0,150)
        ax.set_yticks(np.arange(0,151,30))
        
        ax = axs[2,0]
        err = mag2db((mismatch_ideal.s- mismatch_cal_solr.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        err = mag2db((mismatch_ideal.s- mismatch_cal_srm.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('S11 error (dB)')
        ax.set_yticks(np.arange(-80,0.1,20))
        ax.set_ylim(-80,0)
        
        ax = axs[2,1]
        err = mag2db((offsetshort_ideal.s- offsetshort_cal_solr.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='o', markevery=30, markersize=12,
                label='SOLR', linestyle='-')
        err = mag2db((offsetshort_ideal.s- offsetshort_cal_srm.s))[:,0,0] 
        ax.plot(f*1e-9, err, lw=2.5, marker='X', markevery=30, markersize=12,
                label='SRM', linestyle='-')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xticks(np.arange(0,45,5))
        ax.set_xlim(0,40)
        ax.set_ylabel('S11 error (dB)')
        ax.set_yticks(np.arange(-80,0.1,20))
        ax.set_ylim(-80,0)
        
        #fig.savefig('solr_vs_srm_calibrated_duts.pdf', format='pdf', dpi=300, 
        #        bbox_inches='tight', pad_inches = 0)
    
    
    plt.show()
    