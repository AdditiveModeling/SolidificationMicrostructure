import numpy as np
import sys 
import os
import pf_utils as utils

#dimension (only 2D for python at the moment!)
dim = 2

#material parameters, J, cm, K, s
R = 8.314 #gas constant, J/mol*K

#discretization params
dx = 4.6e-6 #spacial division, cm
dt = 0
d = dx/0.94 #interfacial thickness

#TDB parameters, lists are used for N-component models since we dont know how many variables we need
W = [] #Well size matrix, term for every pair of different phases (for n phases, there are (n^2-n)/2 unique terms)
kij = [] #Order mobility couples, matrix for each pair of phases
D = [] #Diffusion in each phase, cm^2/s. One per phase. Assumed to be constant amongst all species (no Kirkendall effect!)
v_m = 0 #Molar volume, cm^3/mol. Currently considered constant, model does not deal with volume mismatch
M_qmax = [] #orientational mobility, 1/(s*J), term for each phase
H = [] #Orientational Interface energy, J/(K*cm). Term for each phase
ebar2 = [] #epsilon^2 interfacial energy matrix, term for every pair of different phases (for n phases, there are (n^2-n)/2 unique terms)
y_e = [] #anisotropy matrix, term for every pair of different phases (for n phases, there are (n^2-n)/2 unique terms)

#fields, to allow for access outside the simulate function (debugging!)
phi = [] #multiple phases for multiorder model!
c = [] #multiple components for n-component model!
q1 = 0 ###################### Make this an array as well! would make it easier to transition to 3d! #########################
q4 = 0
T = 0

def init_tdb_vars(tdb):
    """
    Modifies the global vars which are parameters for the engine. Called from the function utils.preinitialize
    Returns True if variables are loaded successfully, False if certain variables dont exist in the TDB
    If false, preinitialize will print an error saying the TDB doesn't have enough info to run the sim
    """
    global L, T_M, S, B, W, M, D_S, D_L, v_m, M_qmax, H, ebar, eqbar, dt, y_e
    comps = utils.components
    try:
        L = [] #latent heats, J/cm^3
        T_M = [] #melting temperatures, K
        S = [] #surface energies, J/cm^2
        B = [] #linear kinetic coefficients, cm/(K*s)
        W = [] #Well size
        M = [] #Order mobility coefficient
        T = tdb.symbols[comps[0]+"_L"].free_symbols.pop()
        for i in range(len(comps)):
            L.append(utils.npvalue(T, comps[i]+"_L", tdb))
            T_M.append(utils.npvalue(T, comps[i]+"_TM", tdb))
            S.append(utils.npvalue(T, comps[i]+"_S", tdb))
            B.append(utils.npvalue(T, comps[i]+"_B", tdb))
            W.append(3*S[i]/(np.sqrt(2)*T_M[i]*d))
            M.append(T_M[i]*T_M[i]*B[i]/(6*np.sqrt(2)*L[i]*d)/1574.)
        D_S = utils.npvalue(T, "D_S", tdb)
        D_L = utils.npvalue(T, "D_L", tdb)
        v_m = utils.npvalue(T, "V_M", tdb)
        M_qmax = utils.npvalue(T, "M_Q", tdb)
        H = utils.npvalue(T, "H", tdb)
        y_e = utils.npvalue(T, "Y_E", tdb)
        ebar = np.sqrt(6*np.sqrt(2)*S[1]*d/T_M[1])
        eqbar = 0.5*ebar
        dt = dx*dx/5./D_L/8
        #print(L, T_M, S, B)
        #print(D_S, D_L, v_m, M_qmax, H, dt)
        return True
    except:
        return False
    
def simulate_nc(data_path, initialStep, steps, initT, gradT, dTdt):
    global phi, c, q1, q4, T
    if not os.path.isfile(utils.root_folder+"/data/"+data_path+"/info.txt"):
        print("Simulation has not been initialized yet - aborting engine!")
        return
    
    info = open(utils.root_folder+"/data/"+data_path+"/info.txt", 'a')
    info.write("  New Simulation Run:\n")
    info.write("    Initial Step: "+str(initialStep)+"\n")
    info.write("    Number of steps to run: "+str(steps)+"\n")
    info.write("    Initial Temperature on left edge: "+str(initT)+"\n")
    info.write("    Temperature gradient (K/cell): "+str(gradT)+"\n")
    info.write("    Change in temperature over time (K/time_step): "+str(dTdt)+"\n\n")
    info.close()
    
    nbc = utils.get_nbcs_for_sim(data_path)
    #check to make sure correct TDB isn't already loaded, to avoid having to reinitialize the ufuncs in pycalphad
    if not utils.tdb_path == utils.get_tdb_path_for_sim(data_path):
        utils.tdb_path = utils.get_tdb_path_for_sim(data_path)
        utils.load_tdb(utils.tdb_path)
    
    #load arrays. As of multicomponent model, c is a list of arrays, one per independent component (N-1 total, for an N component model)
    step, phi, c, q1, q4 = utils.loadArrays_nc(data_path, initialStep)
    shape = phi.shape #get the shape of the simulation region from one of the arrays
    
    #temperature
    T = (initT+0.0)*np.ones(shape) #convert T to double just in case
    T += np.linspace(0, gradT*shape[1], shape[1])
    T += step*dTdt

    for i in range(steps):
        step += 1
        g = utils._g(phi)
        h = utils._h(phi)
        m = 1-h;
        M_q = 1e-6 + (M_qmax-1e-6)*m
    
        lq1 = utils.grad2(q1, dx, dim)
        lq4 = utils.grad2(q4, dx, dim)
    
        #additional interpolating functions
        p = phi*phi
        hprime = utils._hprime(phi)
        gprime = utils._gprime(phi)
    
        #quaternion gradient terms
        gq1l = utils.grad_l(q1, dx, dim)
        gq4l = utils.grad_l(q4, dx, dim)
        gqsl = []
        for j in range(dim):
            gqsl.append(gq1l[j]*gq1l[j]+gq4l[j]*gq4l[j])
    
        gq1r = utils.grad_r(q1, dx, dim)
        gq4r = utils.grad_r(q4, dx, dim)
    
        gqsr = []
        for j in range(dim):
            gqsr.append(np.roll(gqsl[j], -1, j))
    
        gqs = (gqsl[0]+gqsr[0])/2
        for j in range(1, dim):
            gqs += (gqsl[j]+gqsr[j])/2
        rgqs_0 = np.sqrt(gqs)
    
        gphi = utils.grad_l(phi, dx, 2)
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
                
        #compute values from tdb
        G_L, dGLdc = utils.compute_tdb_energy_nc(T, c, "LIQUID")
        G_S, dGSdc = utils.compute_tdb_energy_nc(T, c, "FCC_A1")
        
        #change in c1, c2
        M_c = []
        dFdc = []
        deltac = []
        #find the standard deviation as an array 
        std_c=np.sqrt(np.absolute(2*R*T/v_m))
        for j in range(len(c)):
            #find the actual random noise
            noise_c=np.random.normal(0, std_c, phi.shape)
            M_c.append(v_m*c[j]*(D_S+m*(D_L-D_S))/R/1574.)
            #add the change in noise inside the functional
            dFdc.append((dGSdc[j] + m*(dGLdc[j]-dGSdc[j]))/v_m + (W[j]-W[len(c)])*g*T+noise_c)
        for j in range(len(c)):
            deltac.append(utils.divagradb(M_c[j]*(1-c[j]), dFdc[j], dx, dim))
            for k in range(len(c)):
                if not (j == k):
                    deltac[j] -= utils.divagradb(M_c[j]*c[k], dFdc[k], dx, dim)
        
        #change in phi
        divTgradphi = utils.divagradb(T, phi, dx, dim)
            
        #compute overall order mobility, from order mobility coefficients
        c_N = 1-np.sum(c, axis=0)
        M_phi = c_N*M[len(c)]
        for j in range(len(c)):
            M_phi += c[j]*M[j]
                
        #compute well size term for N-components
        well = c_N*W[len(c)]
        for j in range(len(c)):
            well += c[j]*W[j]
        well *= (T*gprime)
            
        psix3 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
        psiy3 = vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
        pf_comp_x = 8*y_e*T*((2*a2_b2*psix3 + 2*ab2*psiy3)/vertex_centered_mgphi2 - vertex_averaged_gphi[0]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
        pf_comp_x = (np.roll(pf_comp_x, -1, 0) - pf_comp_x)/dx
        pf_comp_x = (np.roll(pf_comp_x, -1, 1) + pf_comp_x)/2.
        pf_comp_y = 8*y_e*T*((2*a2_b2*psiy3 - 2*ab2*psix3)/vertex_centered_mgphi2 - vertex_averaged_gphi[1]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
        pf_comp_y = (np.roll(pf_comp_y, -1, 1) - pf_comp_y)/dx
        pf_comp_y = (np.roll(pf_comp_y, -1, 0) + pf_comp_y)/2.
        deltaphi = M_phi*(ebar*ebar*((1-3*y_e)*divTgradphi + pf_comp_x + pf_comp_y)-30*g*(G_S-G_L)/v_m-well-4*H*T*phi*rgqs_0*1574.)
            
        #old noise from Warren1995:
        #randArray = 2*np.random.random_sample(shape)-1
        #alpha = 0.3
        #deltaphi += M_phi*alpha*randArray*(16*g)*(30*g*(G_S-G_L)/v_m+well)
            
        #noise in phi, based on Langevin Noise
        std_phi=np.sqrt(np.absolute(2*R*M_phi*T/v_m))
        noise_phi=np.random.normal(0, std_phi, phi.shape)
        deltaphi += noise_phi
        
        #changes in q, part 1
        dq_component = 2*H*T*p
    
        gaq1 = utils.gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
        gaq4 = utils.gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)
        
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
        for j in range(len(c)):
            c[j] += deltac[j]*dt
        phi += deltaphi*dt
        q1 += deltaq1*dt
        q4 += deltaq4*dt
        utils.applyBCs_nc(phi, c, q1, q4, nbc)
        T += dTdt
        if(i%10 == 0):
            q1, q4 = utils.renormalize(q1, q4)
        
        #This code segment prints the progress after every 5% of the simulation is done (for convenience)
        if(steps > 19):
            if(i%(steps/20) == 0):
                print(str(5*i/(steps/20))+"% done...")
    
        #This code segment saves the arrays every 500 steps, and adds nuclei
        #note: nuclei are added *before* saving the data, so stray nuclei may be found before evolving the system
        if(step%500 == 0):
            #find the stochastic nucleation critical probabilistic cutoff
            #attn -- Q and T_liq are hard coded parameters for Ni-10%Cu
            Q0=8*10**5 #activation energy of migration
            T_liq=1697 #Temperature of Liquidus (K)
            J0,p11=utils.find_Pn(T_liq, T, Q0, dt) 
            #print(J0)
            phi, q1, q4=utils.add_nuclei(phi, q1, q4, p11, len(phi))
            utils.saveArrays_nc(data_path, step, phi, c, q1, q4)

    print("Done")
    
def simulate_ncnp(data_path, initialStep, steps, initT, gradT, dTdt):
    global phi, c, q1, q4, T
    if not os.path.isfile(utils.root_folder+"/data/"+data_path+"/info.txt"):
        print("Simulation has not been initialized yet - aborting engine!")
        return
    
    info = open(utils.root_folder+"/data/"+data_path+"/info.txt", 'a')
    info.write("  New Simulation Run:\n")
    info.write("    Initial Step: "+str(initialStep)+"\n")
    info.write("    Number of steps to run: "+str(steps)+"\n")
    info.write("    Initial Temperature on left edge: "+str(initT)+"\n")
    info.write("    Temperature gradient (K/cell): "+str(gradT)+"\n")
    info.write("    Change in temperature over time (K/time_step): "+str(dTdt)+"\n\n")
    info.close()
    
    nbc = utils.get_nbcs_for_sim(data_path)
    #check to make sure correct TDB isn't already loaded, to avoid having to reinitialize the ufuncs in pycalphad
    if not utils.tdb_path == utils.get_tdb_path_for_sim(data_path):
        utils.tdb_path = utils.get_tdb_path_for_sim(data_path)
        utils.load_tdb(utils.tdb_path)
    
    #load arrays. As of multicomponent model, c is a list of arrays, one per independent component (N-1 total, for an N component model)
    step, phi, c, q1, q4 = utils.loadArrays_ncnp(data_path, initialStep)
    shape = q1.shape #get the shape of the simulation region from one of the arrays
    
    #temperature ### Only linear gradients supported here right now, but any T array could be used in place of this! ###
    T = (initT+0.0)*np.ones(shape) #convert T to double just in case
    T += np.linspace(0, gradT*shape[1], shape[1])
    T += step*dTdt

    for i in range(steps):
        step += 1
        g = utils._g(phi)
        gp = []
        h = []
        hp = []
        p = []
        dTe2dphi = []
        M_q = np.zeros(shape)
        for j in range(len(phi)):
            h.append(utils._h(phi[j]))
            hp.append(utils._hprime(phi[j]))
            M_q = (M_q[j])*h[j]
            p.append(phi[j]**2)
            gp.append(utils._gprime(phi, j))
            dTe2dphi.append(utils.divagradb(T*e2, phi[j], dx, dim))
            
        va_T = va_br(T, dim)
        
        #compute needed terms
        e2, w, de2dpj, de2dpxj, de2dpyj, dwdpj, de2dq1, de2dq4 = doublesums(phi, q1, q4, ebar2, W, y_e, dx)
        
        #quaternion gradient terms
        gq1l = utils.grad_l(q1, dx, dim)
        gq4l = utils.grad_l(q4, dx, dim)
        gqsl = []
        for j in range(dim):
            gqsl.append(gq1l[j]*gq1l[j]+gq4l[j]*gq4l[j])
    
        gq1r = utils.grad_r(q1, dx, dim)
        gq4r = utils.grad_r(q4, dx, dim)
    
        gqsr = []
        for j in range(dim):
            gqsr.append(np.roll(gqsl[j], -1, j))
    
        gqs = (gqsl[0]+gqsr[0])/2
        for j in range(1, dim):
            gqs += (gqsl[j]+gqsr[j])/2
        rgqs_0 = np.sqrt(gqs)
    
        #"clip" the grid: if values are smaller than "smallest", set them equal to "smallest"
        smallest = 1.5
        for j in range(dim):
            gqsl[j] = np.clip(gqsl[j], smallest, np.inf)
            gqsr[j] = np.clip(gqsr[j], smallest, np.inf)
              
        rgqsl = []
        rgqsr = []
        for j in range(dim):
            rgqsl.append(np.sqrt(gqsl[j]))
            rgqsr.append(np.sqrt(gqsr[j]))
                
        #compute values from tdb
        G = []
        dGdc = []
        for j in range(len(phi)):
            G_i, dGidc = utils.compute_tdb_energy_nc(T, c, utils.phases[j])
            G.append(G_i)
            dGdc.append(dGidc)
        
        #change in c1, c2
        M_c = []
        dFdc = []
        deltac = []
        #find the standard deviation as an array 
        std_c=np.sqrt(np.absolute(2*R*T/v_m))
        for j in range(len(c)):
            #find the actual random noise
            noise_c=np.random.normal(0, std_c, phi.shape)
            temp = 0 #mobility
            temp2 = 0 #energy
            for i in range(len(phi)):
                temp += h[i]*D[i]
                temp2 += h[i]*dGdc[i]
            M_c.append(v_m*c[j]*temp/R/1574.)
            #add the change in noise inside the functional
            dFdc.append(temp2/v_m+noise_c)
        for j in range(len(c)):
            deltac.append(utils.divagradb(M_c[j]*(1-c[j]), dFdc[j], dx, dim))
            for k in range(len(c)):
                if not (j == k):
                    deltac[j] -= utils.divagradb(M_c[j]*c[k], dFdc[k], dx, dim)
        
        #change in phi
        divTgradphi = utils.divagradb(T, phi, dx, dim)
            
        #compute overall order mobility, from order mobility coefficients
        c_N = 1-np.sum(c, axis=0)
        M_phi = c_N*M[len(c)]
        for j in range(len(c)):
            M_phi += c[j]*M[j]
                
        #compute well size term for N-components
        well = c_N*W[len(c)]
        for j in range(len(c)):
            well += c[j]*W[j]
        well *= (T*gprime)
        
        dFdphi = []
        for j in range(len(phi)):
            dFdphi.append(0.5*T*de2dpj[j]*smgphi2 + h[j]*G[j]+dwdpj[j]*g[j]*T + w*gp[j]*T+H[j]*T*(2*phi[j])*rgqs_0 - vg_ul(0.5*va_T*de2dpxj*vasmgphi2, dim, 0, dx) - vg_ul(0.5*va_T*de2dpyj*vasmgphi2, dim, 1, dx) - dTe2dphi[j])
        
        deltaphi = []
        for k in range(len(phi)):
            deltaphi.append(0)
            for j in range(len(phi)):
                deltaphi[k] += kij[k][j]*dFdphi[j]
            
        #noise in phi, based on Langevin Noise
        std_phi=np.sqrt(np.absolute(2*R*M_phi*T/v_m))
        noise_phi=np.random.normal(0, std_phi, phi.shape)
        deltaphi += noise_phi
        
        #changes in q, part 1
        dq_component = 0
        for j in range(len(phi)):
            dq_component += p[j]*H[j]
        dq_component *= T
    
        gaq1 = utils.gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
        gaq4 = utils.gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)
        
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
        
        lq1 = utils.grad2(q1, dx, dim)
        lq4 = utils.grad2(q4, dx, dim)
        t1 = eqbar*eqbar*lq1+(gaq1)*1574. + 0.5*T*de2dq1*smgphi2
        t4 = eqbar*eqbar*lq4+(gaq4)*1574. + 0.5*T*de2dq4*smgphi2
        lmbda = (q1*t1+q4*t4)
        deltaq1 = M_q*(t1-q1*lmbda)
        deltaq4 = M_q*(t4-q4*lmbda)
    
        #changes in q
    
    
    
        #apply changes
        for j in range(len(c)):
            c[j] += deltac[j]*dt
        for j in range(len(phi)):
            phi[j] += deltaphi[j]*dt
        q1 += deltaq1*dt
        q4 += deltaq4*dt
        utils.applyBCs_nc(phi, c, q1, q4, nbc)
        T += dTdt
        if(i%10 == 0):
            q1, q4 = utils.renormalize(q1, q4)
        
        #This code segment prints the progress after every 5% of the simulation is done (for convenience)
        if(steps > 19):
            if(i%(steps/20) == 0):
                print(str(5*i/(steps/20))+"% done...")
    
        #This code segment saves the arrays every 500 steps, and adds nuclei
        #note: nuclei are added *before* saving the data, so stray nuclei may be found before evolving the system
        if(step%500 == 0):
            #find the stochastic nucleation critical probabilistic cutoff
            #attn -- Q and T_liq are hard coded parameters for Ni-10%Cu
            Q0=8*10**5 #activation energy of migration
            T_liq=1697 #Temperature of Liquidus (K)
            J0,p11=utils.find_Pn(T_liq, T, Q0, dt) 
            #print(J0)
            phi, q1, q4=utils.add_nuclei(phi, q1, q4, p11, len(phi))
            utils.saveArrays_nc(data_path, step, phi, c, q1, q4)

    print("Done")

#path, _nbc_x, _nbc_y, initialStep, steps, these are the command line arguments

if __name__ == '__main__':
    if(len(sys.argv) == 7):
        data_path = sys.argv[1]
        initial_step = int(sys.argv[2])
        steps = int(sys.argv[3])
        init_T = float(sys.argv[4])
        grad_T = float(sys.argv[5])
        dTdt = float(sys.argv[6])
        utils.tdb_path = utils.get_tdb_path_for_sim(data_path)
        utils.load_tdb(utils.tdb_path)
        init_tdb_vars(utils.tdb)
        print(L, T_M, S, B)
        print(D_S, D_L, v_m, M_qmax, H, dt, y_e)
        simulate_nc(data_path, initial_step, steps, init_T, grad_T, dTdt)
    else:
        print("Error! Needs exactly 6 additional arguments! (data_path, initial_step, steps, init_T, grad_T, dT/dt)")