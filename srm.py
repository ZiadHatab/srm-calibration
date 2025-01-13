"""
@author: Ziad (zi.hatab@gmail.com)

Implementation of the symmetric-reciprocal-match (SRM) calibration method.

Z. Hatab, M. E. Gadringer and W. Bösch, "Symmetric-Reciprocal-Match Method for Vector Network Analyzer Calibration," 
in _IEEE Transactions on Instrumentation and Measurement_, vol. 73, pp. 1-11, 2024,
doi: https://doi.org/10.1109/TIM.2024.3350124, e-print: https://arxiv.org/abs/2309.02886
"""
import warnings
import scipy.optimize
import skrf as rf
import numpy as np
import scipy    # for nonlinear optimization

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

def solve_symmetric_mobius(GammaA, GammaB, lower_rank=False):
    # GammaA : list of input reflection from port-A
    # GammaB : list of input reflection from port-B
    # see equation (14) in the paper
    H = np.array([[-b, -1, b*a, a] for a,b in zip(GammaA,GammaB)])

    if lower_rank:
        E = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        H = H@E

    _,_,vh = np.linalg.svd(H) # compute the SVD
    nullspace = vh[-1,:].conj()
    nullspace = E@nullspace if lower_rank else nullspace  # undo E
    return nullspace.reshape((2,2), order='C')    # get the nullspace and reshape as matrix

def solve_box(W, Gamma, rho=0, port_A=True):
    # if port_A is False, then use port_B
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

def mobius(T, z):
    # mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t11*z + t12)/(t21*z + t22)

def mobius_inv(T, z):
    # inverse mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t12 - t22*z)/(t21*z - t11)

def solve_at_one_freq(symmetric, est_symmetric, reciprocal, est_reciprocal, 
                      reciprocal_GammaA, reciprocal_GammaB, 
                      matchA, matchB, matchA_def, matchB_def, use_half_network, use_symmetric_network=True):
    
    # network measurement
    Mnet = s2t(reciprocal)

    # one-port symmetric standards
    GammaA = symmetric[:,0,0]
    GammaB = symmetric[:,1,1]

    # solve for H = A@P@B@P
    if use_symmetric_network:
        APNinvAinv = solve_symmetric_mobius(GammaA, [mobius(Mnet@p,x) for x in GammaB], lower_rank=True)
        H = APNinvAinv@Mnet@p
    else:
        H  = solve_symmetric_mobius(GammaA, GammaB)
    Hinv = np.linalg.inv(H)
    
    # solve for fake thru; depending what was provided. if both ports are given, compute average
    if np.isnan(reciprocal_GammaA[0]):
        if use_symmetric_network:
            PBinvNinvPNBP = solve_symmetric_mobius([mobius_inv(Mnet@p,x) for x in GammaA], reciprocal_GammaB, lower_rank=True)
            Fb = Mnet@p@PBinvNinvPNBP
        else:
            Fb = solve_symmetric_mobius(GammaA, reciprocal_GammaB)
        Fbinv = np.linalg.inv(Fb)
        M_thru = Fb@Hinv@Mnet@p@Fbinv@H@p if use_half_network else Mnet@p@Fbinv@H@p

    elif np.isnan(reciprocal_GammaB[0]):
        if use_symmetric_network:
            ANPNinvAinv = solve_symmetric_mobius(reciprocal_GammaA, [mobius(Mnet@p,x) for x in GammaB], lower_rank=True)
            Fa = ANPNinvAinv@Mnet@p
        else:
            Fa = solve_symmetric_mobius(reciprocal_GammaA, GammaB)
        Fainv = np.linalg.inv(Fa)
        M_thru = H@Fainv@Mnet@p@Hinv@Fa@p if use_half_network else H@Fainv@Mnet

    else:
        if use_symmetric_network:
            ANPNinvAinv = solve_symmetric_mobius(reciprocal_GammaA, [mobius(Mnet@p,x) for x in GammaB], lower_rank=True)
            Fa = ANPNinvAinv@Mnet@p

            PBinvNinvPNBP = solve_symmetric_mobius([mobius_inv(Mnet@p,x) for x in GammaA], reciprocal_GammaB, lower_rank=True)
            Fb = Mnet@p@PBinvNinvPNBP
        else:
            Fa = solve_symmetric_mobius(reciprocal_GammaA, GammaB)
            Fb = solve_symmetric_mobius(GammaA, reciprocal_GammaB)

        Fainv = np.linalg.inv(Fa)
        M_thru_a = H@Fainv@Mnet@p@Hinv@Fa@p if use_half_network else H@Fainv@Mnet
        M_thru_a = M_thru_a/M_thru_a[-1,-1]
        
        Fbinv = np.linalg.inv(Fb)
        M_thru_b = Fb@Hinv@Mnet@p@Fbinv@H@p if use_half_network else Mnet@p@Fbinv@H@p
        M_thru_b = M_thru_b/M_thru_b[-1,-1]
        
        M_thru   = (M_thru_a + M_thru_b)/2
    
    # perform eigen-decomposition
    # port-A
    _,Wa = np.linalg.eig(M_thru@p@Hinv)
    A1 = solve_box(Wa, matchA, rho=matchA_def)
    A2 = solve_box(Wa@p, matchA, rho=matchA_def)
    err1 = abs( np.array([mobius(A1,est) - G for G,est in zip(GammaA, est_symmetric)]) ).sum()
    err2 = abs( np.array([mobius(A2,est) - G for G,est in zip(GammaA, est_symmetric)]) ).sum()
    first = True if err1 < err2 else False
    A = A1 if first else A2
    Wa = Wa if first else Wa@p
    # port-B
    _,Wb = np.linalg.eig((p@Hinv@M_thru).T)
    B1 = solve_box(Wb, matchB, rho=matchB_def, port_A=False)
    B2 = solve_box(Wb@p, matchB, rho=matchB_def, port_A=False)
    err1 = abs( np.array([mobius_inv(p@B1@p,est) - G for G,est in zip(GammaB, est_symmetric)]) ).sum()
    err2 = abs( np.array([mobius_inv(p@B2@p,est) - G for G,est in zip(GammaB, est_symmetric)]) ).sum()
    first = True if err1 < err2 else False
    B = B1 if first else B2
    Wb = Wb if first else Wb@p
    
    # solve for 7th error term
    k = np.sqrt(np.linalg.det(Mnet)/np.linalg.det(A)/np.linalg.det(B))
    err1 = abs( k*A@s2t(est_reciprocal)@B - Mnet ).sum()
    err2 = abs( k*A@s2t(est_reciprocal)@B + Mnet ).sum()
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

def obj(x, *argv):
    """Objective function for fitting model to one-port stanadards.
    x: model parameters  (this is theta in the paper)

    f: all frequency points
    Wa: eigenvectors at port-A
    Wb: eigenvectors at port-B
    meas_p1: measured reflections at port-A (match and some other reflect standard)
    meas_p2: measured reflections at port-B (match and some other reflect standard)
    model: list of model function that describe the one-port device being fitted (e.g., match and some reflect standard)
    num_var: number of variables for each model function (e.g., 5 for match and 3 for reflect standard)
    """
    
    f  = argv[0]
    Wa = argv[1]
    Wb = argv[2]
    meas_p1 = argv[3]
    meas_p2 = argv[4]
    model   = argv[5]
    num_var = argv[6]
    
    X = [ x[num_var[:inx].sum():num_var[:inx].sum()+y] for inx, y in enumerate(num_var) ]
    
    model_data = np.array([m(f,xx) for m,xx in zip(model, X)])
    
    # port-A
    sigma_p1 = []
    for w, Gamma, rho in zip(Wa, meas_p1.T, model_data.T):
        G1 = np.array([[-1, -1, w[0,0]/w[1,0], w[0,0]/w[1,0]],
                       [1, -1, -w[0,1]/w[1,1], w[0,1]/w[1,1]]])
        
        G2 = np.array([[-r, -1, r*g, g] for r,g in zip(rho,Gamma)])
        
        G = np.vstack((G1,G2))
        s = np.linalg.svd(G, compute_uv=False)
        sigma_p1.append(s[3])
    
    # port-B
    sigma_p2 = []
    for w, Gamma, rho in zip(Wb, meas_p2.T, model_data.T):
        G1 = np.array([[-1, -1, w[0,0]/w[1,0], w[0,0]/w[1,0]],
                       [1, -1, -w[0,1]/w[1,1], w[0,1]/w[1,1]]])
        
        G2 = np.array([[-r , 1, -r*g, g] for r,g in zip(rho,Gamma)])
        
        G = np.vstack((G1,G2))
        s = np.linalg.svd(G, compute_uv=False)
        sigma_p2.append(s[3])
    
    return ( np.array(sigma_p1) + np.array(sigma_p2) ).mean()

class SRM:
    """
    symmetric-reciprocal-match calibration method.
    """
    def __init__(self, symmetric, est_symmetric, reciprocal, est_reciprocal, matchA, matchB, matchA_def=None, matchB_def=None, 
                 reciprocal_GammaA=None, reciprocal_GammaB=None, switch_term=None, 
                 model_fit=None, use_symmetric_network=False, use_half_network=False, fit_max_iter=1000):
        """SRM calibration class.

        Parameters
        ----------
        symmetric (list[Network]): 2-port symmetric networks used as calibration standards
        est_symmetric (list[Network]): 1-port estimates of the symmetric standards 
        reciprocal (Network): 2-port reciprocal network used as calibration standard
        est_reciprocal (Network): 2-port estimate of the reciprocal network
        matchA (Network): 1-port match standard at port A
        matchB (Network): 1-port match standard at port B
        matchA_def (Network, optional): Match definition for port A. If not provided, assume zero (perfect match).
        matchB_def (Network, optional): Match definition for port B.
        reciprocal_GammaA (list[Network]): 1-port measurements at port A. Either port A or B provided or both.
        reciprocal_GammaB (list[Network]): 1-port measurements at port B.
        switch_term (list[Network], optional): Forward and reverse switch terms.
        model_fit (list of dict, optional): Settings for match parasitic fitting. Defaults to None.
        fit_max_iter (int, optional): Maximum number of iterations for the model fitting optimization.
        use_half_network (bool, optional): Use half network procedure (see paper).
        use_symmetric_network (bool, optional): Use symmetric network procedure. 
        This will alow you to use at least 2 one-port standards instead of 3.
        NOTE: don't use "use_symmetric_network", as it's not reliable. I just inlcuded it because why not?
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
        self.use_symmetric_network = use_symmetric_network  # allow at least 2 one-port networks if two-port network is symmetric
        self.use_half_network = use_half_network
        self.fit_max_iter = fit_max_iter
        
        # check number of standards provided
        num_symmetric = self.Ssymmetric.shape[0]
        num_recip_Gamma = self.Sreciprocal_GammaA.shape[0]
        if num_symmetric != num_recip_Gamma:
            raise ValueError('Number of one-port symmetric networks must match number of standards in network-load.')
        
        if num_symmetric == 2 and not self.use_symmetric_network:
            warnings.warn('Only two symmetric standards provided, automatically using symmetric network procedure.')
            self.use_symmetric_network = True

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
                                               use_symmetric_network=self.use_symmetric_network,
                                               use_half_network=self.use_half_network)

            A.append(A_)
            B.append(B_)
            k.append(k_)
            Wa.append(Wa_)
            Wb.append(Wb_)
            print(f'Frequency: {f*1e-9:.2f} GHz ... DONE!')
        
        # perform optimization to model fit the match standard
        if self.model_fit:
            print('Running optimization to fit match standard...')
            # model_fit is a list of dictionary containing information on each standard being fitted
            f = self.f
            x0      = []    # initial values
            bounds  = []    # solution bounds
            meas_p1 = []    # measured reflection at port-A
            meas_p2 = []    # measured reflection at port-B
            model   = []    # model function
            num_var = []    # number of variables for each model function
            for item in self.model_fit:
                x0     = x0 + item['initialValues']
                bounds = bounds + item['bounds']
                meas_p1.append(item['meas'][0])
                meas_p2.append(item['meas'][1])
                model.append(item['func'])
                num_var.append(len(item['initialValues']))
            num_var = np.array(num_var)
            meas_p1 = np.array(meas_p1)
            meas_p2 = np.array(meas_p2)
            
            save_sol = [] # save solution from each iteration
            save_iteration_results = lambda xk,convergence: save_sol.append(xk)
            xx = scipy.optimize.differential_evolution(obj, bounds, x0=x0, args=(f,Wa,Wb,meas_p1,meas_p2,model,num_var),
                                                    disp=True, polish=True, maxiter=self.fit_max_iter, 
                                                    strategy='randtobest1bin', init='sobol',
                                                    popsize=10, mutation=(0.1,1.9), recombination=0.9, 
                                                    tol=1e-6,
                                                    updating='deferred', workers=-1, callback=save_iteration_results
                                                    )
            '''
            # use gradient based optimization (can get you the wrong answer for large number of variables)
            save_iteration_results = lambda xk: save_x.append(xk)
            xx = scipy.optimize.minimize(obj, x0=x0, args=(f,Wa,Wb,meas_p1,meas_p2,model,num_var), 
                                         method='Nelder-Mead', bounds=bounds, callback=save_iteration_results,
                                         options={'disp': True}, tol=1e-10)
            '''
            # final solution from the optimization
            model_para_final = [ xx.x[num_var[:inx].sum():num_var[:inx].sum()+y] for inx, y in enumerate(num_var) ]
            # all solutions from the optimization at each iteration (last one is the final solution)
            model_para_all = []
            for x in save_sol:
                model_para_all.append([ x[num_var[:inx].sum():num_var[:inx].sum()+y] for inx, y in enumerate(num_var) ])
            
            self.model_eval  = np.array([m(f,x) for m,x in zip(model, model_para_final)])  # evaluate the model at the final solution
            self.model_para = model_para_final
            self.model_para_all = model_para_all
            # solve for the error terms using the final solution for the match standard
            A = [solve_box(w, m, r, True) for w,m,r in zip(Wa, meas_p1[0], self.model_eval[0])]     # first one is always assumed to be match standard
            B = [solve_box(w, m, r, False) for w,m,r in zip(Wb, meas_p2[0], self.model_eval[0])]
            k_est = k
            k = []
            for a,b,s,ke in zip(A,B,self.Sreciprocal,k_est):
                k_ = np.sqrt(np.linalg.det(s2t(s))/np.linalg.det(a)/np.linalg.det(b))
                k.append( k_ if abs(k_-ke) < abs(k_+ke) else -k_ )

        self.A  = np.array(A)
        self.B  = np.array(B)
        self.X  = np.array([np.kron(b.T, a) for a,b in zip(self.A, self.B)])
        self.k  = np.array(k)
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
        P = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])  # permute matrix
        q = np.array([[0,1],[1,0]])
        S_cal = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s  # swictch term correction
            """
            Correction based on the bilinear fractional transformation.
            R. A. Speciale, "Projective Matrix Transformations in Microwave Network Theory," 
            1981 IEEE MTT-S International Microwave Symposium Digest, Los Angeles, CA, USA, 
            1981, pp. 510-512, doi: 10.1109/MWSYM.1981.1129979
            """
            A = np.array([[x[2,2],x[2,3]],[x[3,2],1]])
            B = np.array([[x[1,1],x[3,1]],[x[1,3],1]])
            Zero = A*0
            E = P.T@np.block([[A*k, Zero],[Zero, q@np.linalg.inv(B)@q]])@P
            E11,E12,E21,E22 = E[:2,:2], E[:2,2:], E[2:,:2], E[2:,2:]
            S_cal.append( np.linalg.inv(s@E21-E11)@(E12-s@E22) )
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