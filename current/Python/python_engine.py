import numpy as np
import sys 
import os

def __h(phi):
    #h function from Dorr2010
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def __hprime(phi):
    #derivative of the h function from Dorr2010, w.r.t phi
    return (30*phi*phi*(1-phi)*(1-phi))

def __g(phi):
    #g function from Warren1995. Similar to that used in Dorr2010 and Granasy2014
    return (phi*phi*(1-phi)*(1-phi))

def __gprime(phi):
    #derivative of g function, w.r.t phi
    return (4*phi*phi*phi - 6*phi*phi +2*phi)

#Numpy vectorized versions of above functions
_h = np.vectorize(__h)
_hprime = np.vectorize(__hprime)
_g = np.vectorize(__g) 
_gprime = np.vectorize(__gprime)

def grad(phi, dx, dim):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        phip = np.roll(phi, -1, i)
        r.append((phip-phim)/(2*dx))
    return r

def grad_l(phi, dx, dim):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        r.append((phi-phim)/(dx))
    return r

def grad_r(phi, dx, dim):
    r = []
    for i in range(dim):
        phip = np.roll(phi, -1, i)
        r.append((phip-phi)/(dx))
    return r

def partial_l(phi, dx, i):
    phim = np.roll(phi, 1, i)
    return (phi-phim)/dx

def partial_r(phi, dx, i):
    phip = np.roll(phi, -1, i)
    return (phip-phi)/dx

def grad2(phi, dx, dim):
    r = np.zeros_like(phi)
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        phip = np.roll(phi, -1, i)
        r += (phip+phim-2*phi)/(dx*dx)
    return r

def divagradb(a, b, dx, dim):
    r = np.zeros_like(b)
    for i in range(dim):
        agradb = ((a + np.roll(a, -1, i))/2)*(np.roll(b, -1, i) - b)/dx
        r += (agradb - np.roll(agradb, 1, i))/dx
    return r

def gaq(gql, gqr, rgqsl, rgqsr, dqc, dx, dim):
    r = np.zeros_like(dqc)
    for i in range(dim):
        r += ((0.5*(dqc+np.roll(dqc, -1, i))*gqr[i]/rgqsr[i])-(0.5*(dqc+np.roll(dqc, 1, i))*gql[i]/rgqsl[i]))/(dx)
    return r

def renormalize(q1, q4):
    q = np.sqrt(q1*q1+q4*q4)
    return q1/q, q4/q

def loadArrays(path, timestep):
    _q1 = np.load(path+'q1_'+str(timestep)+'.npy')
    _q4 = np.load(path+'q4_'+str(timestep)+'.npy')
    _c = np.load(path+'c_'+str(timestep)+'.npy')
    _phi = np.load(path+'phi_'+str(timestep)+'.npy')
    return timestep, _phi, _c, _q1, _q4

def saveArrays(path, timestep, phi, c, q1, q4):
    np.save(path+'phi_'+str(timestep), phi)
    np.save(path+'c_'+str(timestep), c)
    np.save(path+'q1_'+str(timestep), q1)
    np.save(path+'q4_'+str(timestep), q4)
    
def applyBCs(phi, c, q1, q4, nbc):
    if(nbc[0]):
        c[0,:] = c[1,:]
        c[-1,:] = c[-2,:]
        phi[0,:] = phi[1,:]
        phi[-1,:] = phi[-2,:]
        q1[0,:] = q1[1,:]
        q1[-1,:] = q1[-2,:]
        q4[0,:] = q4[1,:]
        q4[-1,:] = q4[-2,:]
    if(nbc[1]):
        c[:,0] = c[:,1]
        c[:,-1] = c[:,-2]
        phi[:,0] = phi[:,1]
        phi[:,-1] = phi[:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
        
def coreSection(array, nbc):
    returnArray = array
    if(nbc[0]):
        returnArray = returnArray[1:-1, :]
    if(nbc[1]):
        returnArray = returnArray[:, 1:-1]
    return returnArray

#path, _nbc_x, _nbc_y, initialStep, steps, these are the command line arguments

if(len(sys.argv) == 8):
    print(sys.argv)
    max = len(sys.argv)
    path = sys.argv[1]
    
    #Neumann boundary conditions for y and x dimensions (due to the way arrays are organized). 
    #if false, its a periodic boundary instead
    nbc = []
    if((sys.argv[2] == '1') or (sys.argv[2] == 'true') or (sys.argv[2] == 'True')):
        nbc.append(True)
    else:
        nbc.append(False)
    if((sys.argv[3] == '1') or (sys.argv[3] == 'true') or (sys.argv[3] == 'True')):
        nbc.append(True)
    else:
        nbc.append(False)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    #dimension (only 2D for python at the moment!)
    dim = 2

    #bcc = 0L = e, fcc = 1S = d
    #material parameters, J, cm, K, s (except for R and Q terms, which use joules)
    M_qmax = 80000000./1574. #maximum mobility of orientation, 1/(s*J)
    H = 1e-11 #interfacial energy term for quaternions, J/(K*cm)

    #material parameters, from Warren1995
    T_mA = 1728. #melting point of nickel
    T_mB = 1358. #melting point of copper
    L_A = 2350. #latent heat of nickel, J/cm3
    L_B = 1728. #latent heat of copper, J/cm3
    s_A = 0.000037 #surface energy of nickel, J/cm2
    s_B = 0.000029 #surface energy of copper, J/cm2
    D_L = 1e-5 #diffusion in liquid, cm2/s
    D_S = 1e-9 #diffusion in solid, cm2/s
    B_A = 0.33 #linear kinetic coefficient of nickel, cm/K/s
    B_B = 0.39 #linear kinetic coefficient of copper, cm/K/s
    v_m = 7.42 #molar volume, cm3/mol
    R = 8.314 #gas constant, J/mol*K
    y_e = 0.12 #anisotropy
    #y_e = 0.0 #no anisotropy
    e_SA = 6395. #enthalpy of solid Nickel at melting point, J/cm3 (Desai1987)
    e_SB = 4023. #enthalpy of solid Copper at melting point, J/cm3 (Arblaster2015)
    C_A = 5.525 #average heat capacity of nickel at melting point, J/cm3/K  (Desai1987)
    C_B = 4.407 #average heat capacity of copper at melting point, J/cm3/K  (Arblaster2015)

    #discretization params
    dx = 4.6e-6 #spacial division, cm
    dt = dx*dx/5./D_L/8
    d = dx/0.94 #interfacial thickness

    #discretization dependent params, since d is constant, we compute them here for now
    ebar = np.sqrt(6*np.sqrt(2)*s_A*d/T_mA) #baseline energy
    eqbar = 0.5*ebar
    W_A = 3*s_A/(np.sqrt(2)*T_mA*d)
    W_B = 3*s_B/(np.sqrt(2)*T_mB*d)
    M_A = T_mA*T_mA*B_A/(6*np.sqrt(2)*L_A*d)/1574.
    M_B = T_mB*T_mB*B_B/(6*np.sqrt(2)*L_B*d)/1574.
    
    step, phi, c, q1, q4 = loadArrays(path, int(sys.argv[4]))
    shape = c.shape
    
    steps = int(sys.argv[5])
    gradT = float(sys.argv[6])
    #temperature
    T = 1574.*np.ones(shape)
    T += np.linspace(0, gradT*shape[1], shape[1])
    dTdt = float(sys.argv[7])
    T += step*dTdt

    for i in range(steps):
        step += 1
        g = _g(phi)
        h = _h(phi)
        m = 1-h;
        M_q = 1e-6 + (M_qmax-1e-6)*m
    
        lq1 = grad2(q1, dx, dim)
        lq4 = grad2(q4, dx, dim)
    
        #this term is to evolve just the orientation, as done before the first real time step in the Dorr paper
        only_orientation = False
    
        if(only_orientation):
            deltac = 0
            deltaphi = 0
            t1 = eqbar*eqbar*lq1
            t4 = eqbar*eqbar*lq4
            lmbda = (q1*t1+q4*t4)
            deltaq1 = M_q*(t1-q1*lmbda)
            deltaq4 = M_q*(t4-q4*lmbda)
        
        else:
            #additional interpolating functions
            p = phi*phi
            hprime = _hprime(phi)
            gprime = _gprime(phi)
    
            #bulk energy terms, using ideal solution model from Warren1995
            H_A = W_A*gprime*T - 30*L_A*(1-T/T_mA)*g
            H_B = W_B*gprime*T - 30*L_B*(1-T/T_mB)*g
    
            #quaternion gradient terms
            gq1l = grad_l(q1, dx, dim)
            gq4l = grad_l(q4, dx, dim)
            gqsl = []
            for j in range(dim):
                gqsl.append(gq1l[j]*gq1l[j]+gq4l[j]*gq4l[j])
    
            gq1r = grad_r(q1, dx, dim)
            gq4r = grad_r(q4, dx, dim)
    
            gqsr = []
            for j in range(dim):
                gqsr.append(np.roll(gqsl[j], -1, j))
    
            gqs = (gqsl[0]+gqsr[0])/2
            for j in range(1, dim):
                gqs += (gqsl[j]+gqsr[j])/2
            rgqs_0 = np.sqrt(gqs)
    
            gphi = grad_l(phi, dx, 2)
            vertex_averaged_gphi = []
            vertex_averaged_gphi.append((gphi[0]+np.roll(gphi[0], 1, 1))/2.)
            vertex_averaged_gphi.append((gphi[1]+np.roll(gphi[1], 1, 0))/2.)
            vertex_averaged_q1 = (q1 + np.roll(q1, 1, 0))/2.
            vertex_averaged_q1 = (vertex_averaged_q1 + np.roll(vertex_averaged_q1, 1, 1))/2.
            vertex_averaged_q4 = (q4 + np.roll(q4, 1, 0))/2.
            vertex_averaged_q4 = (vertex_averaged_q4 + np.roll(vertex_averaged_q4, 1, 1))/2.
            vertex_averaged_T = (T + np.roll(T, 1, 0))/2.
            vertex_averaged_T = (vertex_averaged_T + np.roll(vertex_averaged_T, 1, 1))/2.
        
            a2_b2 = vertex_averaged_q1*vertex_averaged_q1-vertex_averaged_q4*vertex_averaged_q4
            ab2 = 2.*vertex_averaged_q1*vertex_averaged_q4
        
            vertex_centered_gpsi = []
            vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[0] - ab2*vertex_averaged_gphi[1])
            vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[1] + ab2*vertex_averaged_gphi[0])
        
            psi_xxy = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[1]
            psi_xyy = vertex_centered_gpsi[0]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
            psi_xxyy = psi_xxy*vertex_centered_gpsi[1]
        
            vertex_centered_mgphi2 = vertex_averaged_gphi[0]*vertex_averaged_gphi[0] + vertex_averaged_gphi[1]*vertex_averaged_gphi[1]
    
            #"clip" the grid: if values are smaller than "smallest", set them equal to "smallest"
            #also clip the mgphi values to avoid divide by zero errors!
            smallest = 1.5
            for j in range(dim):
                gqsl[j] = np.clip(gqsl[j], smallest, np.inf)
                gqsr[j] = np.clip(gqsr[j], smallest, np.inf)
            
            vertex_centered_mgphi2 = np.clip(vertex_centered_mgphi2, 0.000000001, np.inf)
              
            rgqsl = []
            rgqsr = []
            for j in range(dim):
                rgqsl.append(np.sqrt(gqsl[j]))
                rgqsr.append(np.sqrt(gqsr[j]))
        
            #change in c
            M_C = v_m*c*(1-c)*(D_S+m*(D_L-D_S))/R/1574.
            temp = W_B*g*T+(1-T/T_mB)*(e_SB-C_B*T_mB+m*L_B)-C_B*T*np.log(T/T_mB)+R*T*np.log(c)/v_m - W_A*g*T - (1-T/T_mA)*(e_SA-C_A*T_mA+m*L_A) + C_A*T*np.log(T/T_mA)-R*T*np.log(1-c)/v_m
            deltac = divagradb(M_C, temp, dx, dim)
            #D_C = D_S+m*(D_L-D_S)
            #temp = D_C*v_m*c*(1-c)*(H_B-H_A)/R
            #deltac = divagradb(D_C, c, dx, dim) + divagradb(temp, phi, dx, dim)
        
            #change in phi
            divTgradphi = divagradb(T, phi, dx, dim)
            M_phi = (1-c)*M_A + c*M_B
        
            psix3 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
            psiy3 = vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
            pf_comp_x = 8*y_e*T*((2*a2_b2*psix3 + 2*ab2*psiy3)/vertex_centered_mgphi2 - vertex_averaged_gphi[0]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
            pf_comp_x = (np.roll(pf_comp_x, -1, 0) - pf_comp_x)/dx
            pf_comp_x = (np.roll(pf_comp_x, -1, 1) + pf_comp_x)/2.
            pf_comp_y = 8*y_e*T*((2*a2_b2*psiy3 - 2*ab2*psix3)/vertex_centered_mgphi2 - vertex_averaged_gphi[1]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
            pf_comp_y = (np.roll(pf_comp_y, -1, 1) - pf_comp_y)/dx
            pf_comp_y = (np.roll(pf_comp_y, -1, 0) + pf_comp_y)/2.
            deltaphi = M_phi*(ebar*ebar*((1-3*y_e)*divTgradphi + pf_comp_x + pf_comp_y)-(1-c)*H_A-c*H_B-4*H*T*phi*rgqs_0*1574.)
            randArray = 2*np.random.random_sample(shape)-1
            alpha = 0.3
            deltaphi += M_phi*alpha*randArray*(16*g)*((1-c)*H_A+c*H_B)
        
            #changes in q, part 1
            dq_component = 2*H*T*p
    
            gaq1 = gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
            gaq4 = gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)
        
            q1px = vertex_averaged_q1*vertex_averaged_gphi[0]
            q1py = vertex_averaged_q1*vertex_averaged_gphi[1]
            q4px = vertex_averaged_q4*vertex_averaged_gphi[0]
            q4py = vertex_averaged_q4*vertex_averaged_gphi[1]
        
            t1_temp = (16*ebar*ebar*T*y_e/vertex_centered_mgphi2)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
            t4_temp = (16*ebar*ebar*T*y_e/vertex_centered_mgphi2)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
            cc_t1_temp = (t1_temp + np.roll(t1_temp, -1, 0))/2.
            cc_t1_temp = (cc_t1_temp + np.roll(cc_t1_temp, -1, 1))/2.
            cc_t4_temp = (t4_temp + np.roll(t4_temp, -1, 0))/2.
            cc_t4_temp = (cc_t4_temp + np.roll(cc_t4_temp, -1, 1))/2.
        
            t1 = eqbar*eqbar*lq1+(gaq1)*1574. + cc_t1_temp
            t4 = eqbar*eqbar*lq4+(gaq4)*1574. + cc_t4_temp
            lmbda = (q1*t1+q4*t4)
            deltaq1 = M_q*(t1-q1*lmbda)
            deltaq4 = M_q*(t4-q4*lmbda)
    
        #changes in q
    
    
    
        #apply changes
        c += deltac*dt
        phi += deltaphi*dt
        q1 += deltaq1*dt
        q4 += deltaq4*dt
        applyBCs(c, phi, q1, q4, nbc)
        T += dTdt
        if(i%10 == 0):
            q1, q4 = renormalize(q1, q4)
        
        #This code segment prints the progress after every 5% of the simulation is done (for convenience)
        if(steps > 19):
            if(i%(steps/20) == 0):
                print(str(5*i/(steps/20))+"% done...")
    
        #This code segment saves the arrays every 1000 steps
        if(step%500 == 0):
            saveArrays(path, step, phi, c, q1, q4)

    print("Done")
elif(sys.argv[1] != "-f"): #this happens when file is loaded in jupyter notebook
    print(len(sys.argv))
    print(sys.argv)
    print("Error! Needs exactly 7 additional arguments! (path, nbc_x, nbc_y, initialStep, steps, gradT, dT/dt)") 