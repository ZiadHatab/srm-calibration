"""
@author: Ziad (zi.hatab@gmail.com)

This is an implementation of the multiline calibration algorithm. 
The base algorithm is described in [1]. Additionally, the code has been 
extended to support a thru-free implementation in which additional
standards are used in place of the thru standard, which is described in [2].

[1] Z. Hatab, M. Gadringer and W. Bösch, 
Improving The Reliability of The Multiline TRL Calibration Algorithm," 
2022 98th ARFTG Microwave Measurement Conference (ARFTG), 
Las Vegas, NV, USA, 2022, pp. 1-5, doi: 10.1109/ARFTG52954.2022.9844064.

[2] Z. Hatab, M. E. Gadringer, and W. Bösch, 
"A Thru-free Multiline Calibration," 2023, doi: arxiv
"""

import skrf as rf
import numpy as np

# constants
c0 = 299792458
Q = np.array([[0,0,0,1], [0,-1,0,0], [0,0,-1,0], [1,0,0,0]])
P = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])

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

def sqrt_unwrapped(z):
    '''
    Take the square root of a complex number with unwrapped phase.
    '''
    return np.sqrt(abs(z))*np.exp(0.5*1j*np.unwrap(np.angle(z)))

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

def compute_G_with_takagi(A):
    # implementation of Takagi decomposition to compute the matrix G used to determine the weighting matrix.
    # Singular value decomposition for the Takagi factorization of symmetric matrices
    # https://www.sciencedirect.com/science/article/pii/S0096300314002239
    u,s,vh = np.linalg.svd(A)
    u,s,vh = u[:,:2],s[:2],vh[:2,:]  # low-rank truncated (Eckart-Young-Mirsky theorem)
    phi = np.sqrt( s*np.diag(vh@u.conj()) )
    G = u@np.diag(phi)
    lambd = s[0]*s[1]  # this is the eigenvalue of the weighted eigenvalue problem (squared Frobenius norm of W)
    return G, lambd

def WLS(x,y,w=1):
    # Weighted least-squares for a single parameter estimation
    x = x*(1+0j) # force x to be complex type 
    return (x.conj().dot(w).dot(y))/(x.conj().dot(w).dot(x))

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X_inv, M, lengths, gamma_est, inx=0):
    # gamma = alpha + 1j*beta is determined through linear weighted least-squares    
    lengths = lengths - lengths[inx]
    EX = (X_inv@M)[[0,-1],:]                  # extract z and y columns
    EX = np.diag(1/EX[:,inx])@EX              # normalize to a reference line based on index `inx` (can be any)
    
    del_inx = np.arange(len(lengths)) != inx  # get rid of the reference line (i.e., thru)
    
    # solve for alpha
    l = -2*lengths[del_inx]
    gamma_l = np.log(EX[0,:]/EX[-1,:])[del_inx]
    alpha =  WLS(l, gamma_l.real, Vgl(len(l)+1))
    
    # solve for beta
    l = -lengths[del_inx]
    gamma_l = np.log((EX[0,:] + 1/EX[-1,:])/2)[del_inx]
    n = np.round( (gamma_l - gamma_est*l).imag/np.pi/2 )
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrap
    beta = WLS(l, gamma_l.imag, Vgl(len(l)+1))
    
    return alpha + 1j*beta 

def solve_quadratic(v1, v2, inx, x_est):
    # inx contain index of the unit value and product 
    v12,v13 = v1[inx]
    v22,v23 = v2[inx]
    mask = np.ones(v1.shape, bool)
    mask[inx] = False
    v11,v14 = v1[mask]
    v21,v24 = v2[mask]
    if abs(v12) > abs(v22):  # to avoid dividing by small numbers
        k2 = -v11*v22*v24/v12 + v11*v14*v22**2/v12**2 + v21*v24 - v14*v21*v22/v12
        k1 = v11*v24/v12 - 2*v11*v14*v22/v12**2 - v23 + v13*v22/v12 + v14*v21/v12
        k0 = v11*v14/v12**2 - v13/v12
        c2 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c1 = (1 - c2*v22)/v12
    else:
        k2 = -v11*v12*v24/v22 + v11*v14 + v12**2*v21*v24/v22**2 - v12*v14*v21/v22
        k1 = v11*v24/v22 - 2*v12*v21*v24/v22**2 + v12*v23/v22 - v13 + v14*v21/v22
        k0 = v21*v24/v22**2 - v23/v22
        c1 = np.array([(-k1 - np.sqrt(-4*k0*k2 + k1**2))/(2*k2), (-k1 + np.sqrt(-4*k0*k2 + k1**2))/(2*k2)])
        c2 = (1 - c1*v12)/v22
    x = np.array( [v1*x + v2*y for x,y in zip(c1,c2)] )  # 2 solutions
    mininx = np.argmin( abs(x - x_est).sum(axis=1) )
    return x[mininx]

def multiline_at_one_freq(Slines, lengths, Sreflect, ereff_est, reflect_est, f, 
                     Snetwork, Snetwork_reflect_A, Snetwork_reflect_B, sw=[0,0]):
    
    # Slines: array containing 2x2 S-parameters of each line standard
    # lengths: array containing the lengths of the lines
    # Sreflect: 2x2 S-parameters of the measured reflect standard
    # ereff_est: complex scalar of estimated effective permittivity
    # reflect_est: complex scalar of estimated reflection coefficient of the reflect standard
    # f: scalar, current frequency point
    # sw: 1x2 array holding the forward and reverse switch terms, respectively.
            
    # correct switch term
    Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
    Snetwork = correct_switch_term(Snetwork,sw[0],sw[1]) if np.any(sw) else Snetwork
    
    # measurements
    Mi    = [s2t(x) for x in Slines] # convert to T-parameters        
    M     = np.array([x.flatten('F') for x in Mi]).T
    MinvT = np.array([np.linalg.inv(x).flatten('F') for x in Mi])
          
    ## Compute W from Takagi factorization
    G, lambd = compute_G_with_takagi(MinvT.dot(M[[0,2,1,3]]))
    W = np.conj((G@np.array([[0,1j],[-1j,0]])).dot(G.T))
    
    # estimated gamma to be used to resolve the sign of W
    gamma_est = 2*np.pi*f/c0*np.sqrt(-(ereff_est-1j*np.finfo(float).eps))  # the eps is to ensure positive square-root
    gamma_est = abs(gamma_est.real) + 1j*abs(gamma_est.imag)  # this to avoid sign inconsistencies 
        
    z_est = np.exp(-gamma_est*lengths)
    y_est = 1/z_est
    W_est = (np.outer(y_est,z_est) - np.outer(z_est,y_est)).conj()
    W = -W if abs(W-W_est).sum() > abs(W+W_est).sum() else W # resolve the sign ambiguity
    
    ## Solving the weighted eigenvalue problem
    F = np.dot(M,np.dot(W,MinvT[:,[0,2,1,3]]))  # weighted measurements
    eigval, eigvec = np.linalg.eig(F+lambd*np.eye(4))
    inx = np.argsort(abs(eigval))
    v1 = eigvec[:,inx[0]]
    v2 = eigvec[:,inx[1]]
    v3 = eigvec[:,inx[2]]
    v4 = eigvec[:,inx[3]]
    x1__est = v1/v1[0]
    x1__est[-1] = x1__est[1]*x1__est[2]
    x4_est = v4/v4[-1]
    x4_est[0] = x4_est[1]*x4_est[2]
    x2__est = np.array([x4_est[2], 1, x4_est[2]*x1__est[2], x1__est[2]])
    x3__est = np.array([x4_est[1], x4_est[1]*x1__est[1], 1, x1__est[1]])
    
    # solve quadratic equation for each column
    x1_ = solve_quadratic(v1, v4, [0,3], x1__est)
    x2_ = solve_quadratic(v2, v3, [1,2], x2__est)
    x3_ = solve_quadratic(v2, v3, [2,1], x3__est)
    x4  = solve_quadratic(v1, v4, [3,0], x4_est)
    
    # build the normalized cal coefficients (average the answers from range and null spaces)    
    a12 = (x2_[0] + x4[2])/2
    b21 = (x3_[0] + x4[1])/2
    a21_a11 = (x1_[1] + x3_[3])/2
    b12_b11 = (x1_[2] + x2_[3])/2
    X_  = np.kron([[1,b21],[b12_b11,1]], [[1,a12],[a21_a11,1]])
    
    X_inv = np.linalg.inv(X_)
    
    ## Compute propagation constant
    gamma = compute_gamma(X_inv, M, lengths, gamma_est)
    ereff = -(c0/2/np.pi/f*gamma)**2

    # solve for a11b11
    if np.isnan(Snetwork_reflect_A) and np.isnan(Snetwork_reflect_B):
        # make first line as Thru, i.e., zero length
        # lengths = np.array([x-lengths[0] for x in lengths])
        # solve for a11b11 and k from Thru measurement
        ka11b11,_,_,k = X_inv.dot(M[:,0]).squeeze()
        a11b11 = ka11b11/k
    else:
        Ns = t2s( (X_inv@s2t(Snetwork).flatten('F')).reshape((2,2), order='F') )
        m2 = Ns[0,0]
        m4 = Ns[1,1]
        m5 = Ns[0,1]*Ns[1,0]
        m1 = (Sreflect[0,0] - a12)/(1 - Sreflect[0,0]*a21_a11)
        m3 = (Sreflect[1,1] + b21)/(1 + Sreflect[1,1]*b12_b11)
        m6 = (Snetwork_reflect_A - a12)/(1 - Snetwork_reflect_A*a21_a11)
        m7 = (Snetwork_reflect_B + b21)/(1 + Snetwork_reflect_B*b12_b11)
        a11b11_A = (m1*m2*m4-m1*m5-m6*m1*m4)/(m2-m6)
        a11b11_B = (m3*m2*m4-m3*m5-m7*m3*m2)/(m4-m7)
        a11b11_A = a11b11_B if np.isnan(a11b11_A.real) else a11b11_A
        a11b11_B = a11b11_A if np.isnan(a11b11_B.real) else a11b11_B
        a11b11 = (a11b11_A + a11b11_B)/2

    T  = np.dot(X_inv, s2t(Sreflect, pseudo=True).flatten('F') ).squeeze()
    a11_b11 = -T[2]/T[1]
    a11 = np.sqrt(a11_b11*a11b11)
    b11 = a11b11/a11
    G_cal = ( (Sreflect[0,0] - a12)/(1 - Sreflect[0,0]*a21_a11)/a11 + (Sreflect[1,1] + b21)/(1 + Sreflect[1,1]*b12_b11)/b11 )/2  # average
    if abs(G_cal - reflect_est) > abs(G_cal + reflect_est):
        a11 = -a11
        b11 = -b11
        G_cal = -G_cal
    reflect_est = G_cal
    
    # build the calibration matrix (de-normalize)
    X  = np.dot( X_, np.diag([a11b11, b11, a11, 1]) )
    
    # solve for k
    if not np.isnan(Snetwork_reflect_A) or not np.isnan(Snetwork_reflect_B):
        Xinv = np.linalg.inv(X)
        k2 = np.array([ np.linalg.det( (Xinv@x).reshape((2,2), order='F') ) for x in M.T]).mean()
        k = np.sqrt(k2)
        err1 = abs(np.array([np.exp(-gamma*lengths[-1]),0,0,np.exp(gamma*lengths[-1])])*k - Xinv@M[:,-1]).sum()
        err2 = abs(np.array([np.exp(-gamma*lengths[-1]),0,0,np.exp(gamma*lengths[-1])])*k + Xinv@M[:,-1]).sum()
        k = k if err1 < err2 else -k

    return X, k, ereff, gamma, reflect_est, lambd

class multiline:
    """
    Thru-free multiline calibration.
    """
    
    def __init__(self, lines, line_lengths, reflect, reflect_est=-1, reflect_offset=0, 
                 ereff_est=1+0j, network=None, network_reflect_A=None, 
                 network_reflect_B=None, switch_term=None):
        """
        thru-free multiline initializer.
        """
        
        self.f  = lines[0].frequency.f
        self.Slines = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.Sreflect = reflect.s
        self.reflect_est = reflect_est
        self.reflect_offset = reflect_offset
        self.ereff_est = ereff_est

        self.Snetwork = self.Slines[0] if network is None else network.s
        self.Snetwork_reflect_A = np.nan*self.f if network_reflect_A is None else network_reflect_A.s.squeeze()
        self.Snetwork_reflect_B = np.nan*self.f if network_reflect_B is None else network_reflect_B.s.squeeze()

        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
    
    def run_multiline(self):
        # This runs the standard multiline without uncertainties (very fast).
        gammas  = []
        lambds  = []
        Xs      = []
        ks      = []
        ereff0  = self.ereff_est*(1+0.0j)
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        gamma0  = gamma0*np.sign(gamma0.real) # use positive square root
        reflect_est0 = self.reflect_est*np.exp(-2*gamma0*self.reflect_offset)
        reflect_est = reflect_est0
        
        lengths = self.lengths
        print('\nmultiline in progress:')
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Snetwork = self.Snetwork[inx,:,:]
            Snetwork_reflect_A = self.Snetwork_reflect_A[inx]
            Snetwork_reflect_B = self.Snetwork_reflect_B[inx]
            Sreflect = self.Sreflect[inx,:,:]
            sw = self.switch_term[:,inx]
            
            X,k,ereff0,gamma,reflect_est,lambd = multiline_at_one_freq(Slines, lengths, Sreflect, 
                                                    ereff_est=ereff0, reflect_est=reflect_est, f=f, 
                                                    Snetwork=Snetwork, Snetwork_reflect_A=Snetwork_reflect_A, 
                                                    Snetwork_reflect_B=Snetwork_reflect_B, sw=sw)
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {(f*1e-9).round(4)} GHz done!', end='\r', flush=True)

        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
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
            s21_cal = k*s[1,0]/T_[-1]
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
        return rf.Network(frequency=freq, s=S_cal.squeeze()), S_cal
    
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
    
    def shift_plane(self, d=0):
        '''
        Shift calibration plane by a distance d.
        Negative d value shifts toward port, while positive d value shift away from port.
        For example, if your Thru has a length of L, then d=-L/2 shifts the plane backward to the edges of the Thru.
        '''
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, self.gamma):
            z = np.exp(-g*d)
            KX_new = k*x@np.diag([z**2, 1, 1, 1/z**2])
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])
        self.X = np.array(X_new)
        self.k = np.array(K_new)
    
    def renorm_impedance(self, Z_new, Z0=50):
        '''
        Re-normalize reference calibration impedance. by default, the ref impedance is the characteristic 
        impedance of the line standards (even if you don'y know it!).
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