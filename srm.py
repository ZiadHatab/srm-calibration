"""
@author: Ziad (zi.hatab@gmail.com)

Implementation of the symmetric-reciprocal-match (SRM) calibration method.
"""

import skrf as rf
import numpy as np

# constants
c0 = 299792458
p = np.array([[0,1],[1,0]])

def s2t(S, pseudo=False):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S if pseudo else S/T[1,1]

def sqrt_unwrapped(z):
    '''
    Take the square root of a complex number with unwrapped phase.
    '''
    return np.sqrt(abs(z))*np.exp(0.5*1j*np.unwrap(np.angle(z)))

def solve_symmetric_mobius(GammaA, GammaB):
    # GammaA : list of input reflection from port-A
    # GammaB : list of input reflection from port-B
    # see equation (14) in the paper
    #
    H = np.array([[-b, -1, b*a, a] for a,b in zip(GammaA,GammaB)])
    _,_,vh = np.linalg.svd(H) # compute the SVD
    return vh[-1,:].conj().reshape((2,2), order='C')    # get the nullspace and reshape as matrix

def solve_box(W, Gamma, rho=0, port_A=True):
    # if port_A is False, then use port_B
    #
    w11 = W[0,0]/W[1,0]
    w12 = W[0,1]/W[1,1]
    if port_A:
        H = np.array([[-1, -1, w11, w11],
                      [ 1, -1, -w12, w12],
                      [-rho, -1, rho*Gamma, Gamma]])
    else:
        H = np.array([[-1, -1, w11, w11],
                      [ 1, -1, -w12, w12],
                      [-rho , 1, -rho*Gamma, Gamma]])
    _,_,vh = np.linalg.svd(H) # compute the SVD
    nullspace = vh[-1,:].conj()
    nullspace = nullspace/nullspace[-1]
    if port_A:
        return nullspace.reshape((2,2), order='C')
    else:
        return nullspace.reshape((2,2), order='F')

def input_reflection_l2r(T, G):
    return (T[0,1]+T[0,0]*G)/(T[1,1]+T[1,0]*G)

def inverse_reflection_l2r(T, G):
    return (G*T[1,1]-T[0,1])/(T[0,0]-T[1,0]*G)

def input_reflection_r2l(T, G):
    return (T[0,0]*G-T[1,0])/(T[1,1]-T[0,1]*G)

def inverse_reflection_r2l(T, G):
    return (G*T[1,1]+T[1,0])/(T[0,0]+T[0,1]*G)

def solve_at_one_freq(symmetric, est_symmetric, reciprocal, est_reciprocal, 
                      reciprocal_GammaA, reciprocal_GammaB, 
                      matchA, matchB, matchA_def, matchB_def, use_half_network):
    # measurements
    
    H  = solve_symmetric_mobius(symmetric[:,0,0], symmetric[:,1,1])
    M_net = s2t(reciprocal)
    Hinv = np.linalg.inv(H)
    
    # solve for fake thru; depending what was provided. if both ports are given, we compute average
    if np.isnan(reciprocal_GammaA[0]):
        Fb = solve_symmetric_mobius(symmetric[:,0,0], reciprocal_GammaB)
        Fbinv = np.linalg.inv(Fb)
        M_thru = Fb@Hinv@M_net@p@Fbinv@H@p if use_half_network else M_net@p@Fbinv@H@p
    elif np.isnan(reciprocal_GammaB[0]):
        Fa = solve_symmetric_mobius(reciprocal_GammaA, symmetric[:,1,1])
        Fainv = np.linalg.inv(Fa)
        M_thru = H@Fainv@M_net@p@Hinv@Fa@p if use_half_network else H@Fainv@M_net
    else:
        Fa = solve_symmetric_mobius(reciprocal_GammaA, symmetric[:,1,1])
        Fainv = np.linalg.inv(Fa)
        M_thru_a = H@Fainv@M_net@p@Hinv@Fa@p if use_half_network else H@Fainv@M_net
        M_thru_a = M_thru_a/M_thru_a[-1,-1]
        
        Fb = solve_symmetric_mobius(symmetric[:,0,0], reciprocal_GammaB)
        Fbinv = np.linalg.inv(Fb)
        M_thru_b = Fb@Hinv@M_net@p@Fbinv@H@p if use_half_network else M_net@p@Fbinv@H@p
        M_thru_b = M_thru_b/M_thru_b[-1,-1]
        
        M_thru   = (M_thru_a + M_thru_b)/2
        
    _,Wa = np.linalg.eig(M_thru@p@Hinv)
    A1 = solve_box(Wa, matchA, rho=matchA_def)
    A2 = solve_box(Wa@p, matchA, rho=matchA_def)
    err1 = abs( np.array([input_reflection_l2r(A1,est) - G for G,est in zip(symmetric[:,0,0], est_symmetric)]) ).sum()
    err2 = abs( np.array([input_reflection_l2r(A2,est) - G for G,est in zip(symmetric[:,0,0], est_symmetric)]) ).sum()
    first = True if err1 < err2 else False
    A = A1 if first else A2
    Wa = Wa if first else Wa@p
    
    _,Wb = np.linalg.eig((p@Hinv@M_thru).T)
    B1 = solve_box(Wb, matchB, rho=matchB_def, port_A=False)
    B2 = solve_box(Wb@p, matchB, rho=matchB_def, port_A=False)
    err1 = abs( np.array([input_reflection_r2l(B1,est) - G for G,est in zip(symmetric[:,1,1], est_symmetric)]) ).sum()
    err2 = abs( np.array([input_reflection_r2l(B2,est) - G for G,est in zip(symmetric[:,1,1], est_symmetric)]) ).sum()
    first = True if err1 < err2 else False
    B = B1 if first else B2
    Wb = Wb if first else Wb@p
        
    k = np.sqrt(np.linalg.det(s2t(reciprocal))/np.linalg.det(A)/np.linalg.det(B))
    err1 = abs( k*A@s2t(est_reciprocal)@B - s2t(reciprocal) ).sum()
    err2 = abs( k*A@s2t(est_reciprocal)@B + s2t(reciprocal) ).sum()
    k  = k if err1 < err2 else -k
    
    return A, B, k, Wa, Wb

def correct_switch_term(S, GF, GR):
    '''
    correct switch terms of measured S-parameters at a single frequency point
    GF: forward (sourced by port-1)
    GR: reverse (sourced by port-2)
    '''
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    return S_new

class SRM:
    """
    symmetric-reciprocal-match calibration method.
    """
    def __init__(self, symmetric, est_symmetric, reciprocal, est_reciprocal, matchA, matchB, matchA_def=None, matchB_def=None, 
                 reciprocal_GammaA=None, reciprocal_GammaB=None, switch_term=None, model_fit=None, use_half_network=False):
        """
        SRM initializer
        
        Parameters
        --------------
        symmetric : list of :class:`~skrf.network.Network`
             2-port networks
        
        est_symmetric : list of :class:`~skrf.network.Network`
             1-port networks

        reciprocal :class:`~skrf.network.Network`
             2-port transmissive network.
        
        est_reciprocal :class:`~skrf.network.Network`
             2-port transmissive network.
        
        reciprocal_GammaA : list of :class:`~skrf.network.Network`
             1-port networks of measurement of network-load standards at port-A.

        reciprocal_GammaB : list of :class:`~skrf.network.Network`
             1-port networks of measurement of network-load standards at port-B.

        matchA :class:`~skrf.network.Network`
             1-port network match.

        matchB :class:`~skrf.network.Network`
             1-port network match.
        
        matchA_def :class:`~skrf.network.Network`
             1-port network match definition.

        matchB_def :class:`~skrf.network.Network`
             1-port network match definition.
        
        model_fit :dict
             allow automatic determination of the parasitics of the match standard. The user provides the equivalent circuit to be fitted.

        use_half_network: Boolean
            specify the SRM calibration mode to use full or half networks for the network-load standards.
        
        switch_term : list of :class:`~skrf.network.Network`
            list of 1-port networks. Holds 2 elements:
                1. network for forward switch term.
                2. network for reverse switch term.
        """
        self.f  = symmetric[0].frequency.f
        self.Ssymmetric  = np.array([x.s.squeeze() for x in symmetric])
        self.Sest_symmetric  = np.array([x.s.squeeze() for x in est_symmetric])
        self.Sreciprocal = reciprocal.s.squeeze()
        self.Sest_reciprocal = est_reciprocal.s.squeeze()
        self.SmatchA  = matchA.s.squeeze()
        self.SmatchB  = matchB.s.squeeze()
        self.SmatchA_def = self.SmatchA*0 if matchA_def is None else matchA_def.s.squeeze()
        self.SmatchB_def = self.SmatchB*0 if matchB_def is None else matchB_def.s.squeeze()
        
        if reciprocal_GammaA is None:
            self.Sreciprocal_GammaB = np.array([x.s.squeeze() for x in reciprocal_GammaB])
            self.Sreciprocal_GammaA = self.Sreciprocal_GammaB.copy()*np.nan
        elif reciprocal_GammaB is None:
            self.Sreciprocal_GammaA = np.array([x.s.squeeze() for x in reciprocal_GammaA])
            self.Sreciprocal_GammaB = self.Sreciprocal_GammaA.copy()*np.nan
        else:
            self.Sreciprocal_GammaA = np.array([x.s.squeeze() for x in reciprocal_GammaA])
            self.Sreciprocal_GammaB = np.array([x.s.squeeze() for x in reciprocal_GammaB])
        
        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
            
        self.model_fit = model_fit
        self.use_half_network = use_half_network
        
    def run(self):
        # This runs the calibration procedure
        A = []
        B = []
        k = []
        Wa = []
        Wb = []
        print('\nSRM is running...')
        for inx, f in enumerate(self.f):
            symmetric  = self.Ssymmetric[:,inx,:,:]
            est_symmetric  = self.Sest_symmetric[:,inx]
            reciprocal = self.Sreciprocal[inx,:,:]
            est_reciprocal = self.Sest_reciprocal[inx,:,:]
            reciprocal_GammaA = self.Sreciprocal_GammaA[:,inx]
            reciprocal_GammaB = self.Sreciprocal_GammaB[:,inx]
            matchA  = self.SmatchA[inx]
            matchB  = self.SmatchB[inx]
            matchA_def = self.SmatchA_def[inx]
            matchB_def = self.SmatchB_def[inx]
            
            sw = self.switch_term[:,inx]
            reciprocal = correct_switch_term(reciprocal,sw[0],sw[1]) # correct switch term
                        
            A_, B_, k_, Wa_, Wb_ = solve_at_one_freq(symmetric=symmetric, est_symmetric=est_symmetric, 
                                               reciprocal=reciprocal, est_reciprocal=est_reciprocal, 
                                               reciprocal_GammaA=reciprocal_GammaA, reciprocal_GammaB=reciprocal_GammaB,
                                               matchA=matchA, matchB=matchB, matchA_def=matchA_def, matchB_def=matchB_def,
                                               use_half_network=self.use_half_network)

            A.append(A_)
            B.append(B_)
            k.append(k_)
            Wa.append(Wa_)
            Wb.append(Wb_)
            print(f'Frequency: {f*1e-9:.2f} GHz ... DONE!')
            
        self.A = np.array(A)
        self.B = np.array(B)
        self.X = np.array([np.kron(b.T, a) for a,b in zip(self.A, self.B)])
        self.k = np.array(k)
        self.Wa = np.array(Wa)
        self.Wb = np.array(Wb)
        self.error_coef() # compute the 12 error terms
    
    def apply_cal(self, NW, left=True):
        '''
        Apply calibration to a 1-port or 2-port network.
        NW:   the network to be calibrated (1- or 2-port).
        left: boolean: define which port to use when 1-port network is given. If left is True, left port is used; otherwise right port is used.
        '''
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)
        # apply cal
        S_cal = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s
            xinv = np.linalg.pinv(x)
            M_ = np.array([-s[0,0]*s[1,1]+s[0,1]*s[1,0], -s[1,1], s[0,0], 1])
            T_ = xinv@M_
            s21_cal = k*s[1,0]/T_[-1] + np.finfo(float).eps
            T_ = T_/T_[-1]
            S_cal.append([[T_[2], (T_[0]-T_[2]*T_[1])/s21_cal],[s21_cal, -T_[1]]])
        S_cal = np.array(S_cal).squeeze()
        freq  = NW.frequency
        
        # revert to 1-port device if the input was a 1-port device
        if nports < 2:
            if left: # left port
                S_cal = S_cal[:,0,0]
            else:  # right port
                S_cal = S_cal[:,1,1]
        return rf.Network(frequency=freq, s=S_cal.squeeze())
    
    def error_coef(self):
        '''
        [4] R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 50th ARFTG Conference Digest, 1997, pp. 115-126.
        [5] Dunsmore, J.P.. Handbook of Microwave Component Measurements: with Advanced VNA Techniques.. Wiley, 2020.

        The following list includes the full error term abbreviations. In reference [4], Marks used the abbreviations without providing their full forms, 
        which can be challenging to understand for those unfamiliar with VNA calibration terminology. 
        For a comprehensive understanding of VNAs, I recommend consulting the book by Dunsmore [5], where all the terms are listed in full.
        
        Left port error terms (forward direction):
        EDF: forward directivity
        ESF: forward source match
        ERF: forward reflection tracking
        ELF: forward load match
        ETF: forward transmission tracking
        EXF: forward crosstalk
        
        Right port error terms (reverse direction):
        EDR: reverse directivity
        ESR: reverse source match
        ERR: reverse reflection tracking
        ELR: reverse load match
        ETR: reverse transmission tracking
        EXR: reverse crosstalk
        
        Switch terms:
        GF: forward switch term
        GR: reverse switch term

        NOTE: the k in my notation is equivalent to Marks' notation [4] by this relationship: k = (beta/alpha)*(1/ERR).
        '''

        self.coefs = {}
        # forward 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDF =  self.X[:,2,3]
        ESF = -self.X[:,3,2]
        ERF =  self.X[:,2,2] - self.X[:,2,3]*self.X[:,3,2]
        
        # reverse 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDR = -self.X[:,1,3]
        ESR =  self.X[:,3,1]
        ERR =  self.X[:,1,1] - self.X[:,3,1]*self.X[:,1,3]
        
        # switch terms
        GF = self.switch_term[0]
        GR = self.switch_term[1]

        # remaining forward terms
        ELF = ESR + ERR*GF/(1-EDR*GF)  # eq. (36) in [4].
        ETF = 1/self.k/(1-EDR*GF)      # eq. (38) in [4], after substituting eq. (36) in eq. (38) and simplifying.
        EXF = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # remaining reverse terms
        ELR = ESF + ERF*GR/(1-EDF*GR)    # eq. (37) in [4].
        ETR = self.k*ERR*ERF/(1-EDF*GR)  # eq. (39) in [4], after substituting eq. (37) in eq. (39) and simplifying.
        EXR = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # forward direction
        self.coefs['EDF'] = EDF
        self.coefs['ESF'] = ESF
        self.coefs['ERF'] = ERF
        self.coefs['ELF'] = ELF
        self.coefs['ETF'] = ETF
        self.coefs['EXF'] = EXF
        self.coefs['GF']  = GF

        # reverse direction
        self.coefs['EDR'] = EDR
        self.coefs['ESR'] = ESR
        self.coefs['ERR'] = ERR
        self.coefs['ELR'] = ELR
        self.coefs['ETR'] = ETR
        self.coefs['EXR'] = EXR
        self.coefs['GR']  = GR

        # consistency check between 8-terms and 12-terms model. Based on eq. (35) in [4].
        # This should equal zero, otherwise there is inconsistency between the models (can arise from bad switch term measurements).
        self.coefs['check'] = abs( ETF*ETR - (ERR + EDR*(ELF-ESR))*(ERF + EDF*(ELR-ESF)) )
        return self.coefs 

    def reciprocal_ntwk(self):
        '''
        Return left and right error-boxes as skrf networks, assuming they are reciprocal.
        '''
        freq = rf.Frequency.from_f(self.f, unit='hz')
        freq.unit = 'ghz'

        # left error-box
        S11 = self.coefs['EDF']
        S22 = self.coefs['ESF']
        S21 = sqrt_unwrapped(self.coefs['ERF'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        left_ntwk = rf.Network(s=S, frequency=freq, name='Left error-box')
        
        # right error-box
        S11 = self.coefs['EDR']
        S22 = self.coefs['ESR']
        S21 = sqrt_unwrapped(self.coefs['ERR'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        right_ntwk = rf.Network(s=S, frequency=freq, name='Right error-box')
        right_ntwk.flip()
        return left_ntwk, right_ntwk
    
    def renorm_impedance(self, Z_new, Z0=50):
        '''
        Re-normalize reference calibration impedance.
        Z_new: new ref. impedance (can be array if frequency dependent)
        Z0: old ref. impedance (can be array if frequency dependent)
        '''
        # ensure correct array dimensions if scalar is given (frequency independent).
        N = len(self.k)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, G):
            KX_new = k*x@np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2)
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.k = np.array(K_new)


# EOF