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
kijbar = [] #Order mobility couples, matrix for each pair of phases. Must be combined with phase information to compute actual mobilities
D = [] #Diffusion in each phase, cm^2/s. One per phase. Assumed to be constant amongst all species (no Kirkendall effect!)
v_m = 0 #Molar volume, cm^3/mol. Currently considered constant, model does not deal with volume mismatch
M_qmax = [] #orientational mobility, 1/(s*J), term for each phase
H = [] #Orientational Interface energy, J/(K*cm). Term for each phase
ebar2 = [] #epsilon^2 interfacial energy matrix, term for every pair of different phases (for n phases, there are (n^2-n)/2 unique terms)
eqbar = 0 #interfacial orientation energy term
y_e = [] #anisotropy matrix, term for every pair of different phases (for n phases, there are (n^2-n)/2 unique terms)

#fields, to allow for access outside the simulate function (debugging!)
phi = [] #multiple phases for multiorder model!
c = [] #multiple components for n-component model!
q1 = 0 ###################### Make this an array as well! would make it easier to transition to 3d! #########################
q4 = 0
T = 0
    
def init_tdb_vars_ncnp(tdb):
    """
    Modifies the global vars which are parameters for the engine. Called from the function utils.preinitialize
    This function supports arbitrary number of phases *and* components
    Returns True if variables are loaded successfully, False if certain variables dont exist in the TDB
    If false, preinitialize will print an error saying the TDB doesn't have enough info to run the sim
    """
    global W, kijbar, D, v_m, M_qmax, H, ebar2, eqbar, dt, y_e
    comps = utils.components
    try:
        S = [] #surface energies, J/cm^2
        W = [] #Well size
        kijbar = [] #Order mobility coefficient
        D = []
        ebar2 = [] #interfacial order energy term
        M_qmax = [] #orientation mobility
        H = []
        y_e = []
        T = tdb.symbols["V_M"].free_symbols.pop()
        v_m = utils.npvalue(T, "V_M", tdb)
        for i in range(len(utils.phases)):
            S.append([])
            W.append([])
            kijbar.append([])
            ebar2.append([])
            y_e.append([])
            for j in range(i):
                S_value = utils.npvalue(T, "S_"+utils.phases[j]+"_"+utils.phases[i], tdb)
                S[i].append(S_value)
                S[j].append(S_value)
                W_value = 3*S_value/d/1574.
                W[i].append(W_value)
                W[j].append(W_value)
                kij_value = utils.npvalue(T, "K_"+utils.phases[j]+"_"+utils.phases[i], tdb)
                kijbar[i].append(kij_value)
                kijbar[j].append(kij_value)
                ebar2_value = 6*S_value*d/1574.
                ebar2[i].append(ebar2_value)
                ebar2[j].append(ebar2_value)
                ye_value = utils.npvalue(T, "Y_"+utils.phases[j]+"_"+utils.phases[i], tdb)
                y_e[i].append(ye_value)
                y_e[j].append(ye_value)
            S[i].append(0)
            W[i].append(0)
            kijbar[i].append(0)
            ebar2[i].append(0)
            y_e[i].append(0)
            D.append(utils.npvalue(T, "D_"+utils.phases[i], tdb))
            M_qmax.append(utils.npvalue(T, "MQ_"+utils.phases[i], tdb))
            H.append(utils.npvalue(T, "H_"+utils.phases[i], tdb))
            
        eqbar = 0.5*ebar2[0][1] #interfacial orientation term, here we choose an arbitrary interface to be relative to
        dt = dx*dx/5./np.max(D)/8
        return True
    except Exception as e:
        print(e)
        return False
    
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
    
    #debug var, if true, print out lots of extra information after every iteration
    #warning: only run for a handful of steps if you use this!
    DEBUG = False

    for i in range(steps):
        
        step += 1
        g = utils._g(phi)
        gp = []
        h = []
        hp = []
        p = []
        dTe2dphi = []
        pbp = []
        kij = []
        smgphi2 = np.zeros(shape)
        vasmgphi2 = np.zeros(shape)
        M_q = np.zeros(shape)
        
        #compute needed terms
        e2, w, de2dpj, de2dpxj, de2dpyj, dwdpj, de2dq1, de2dq4 = utils.doublesums(phi, q1, q4, ebar2, W, y_e, dx)
        
        va_T = utils.va_br(T, dim)
        gT = utils.grad(T, dx, dim)
        va_e2 = utils.va_ul(e2, dim)
        e2_x = utils.vg_ul(e2, dim, 0, dx)
        e2_y = utils.vg_ul(e2, dim, 1, dx)
        
        for j in range(len(phi)):
            h.append(utils._h(phi[j]))
            hp.append(utils._hprime(phi[j]))
            M_q += (M_qmax[j])*h[j]
            p.append(phi[j]**2)
            gp.append(utils._gprime(phi, j))
            gphi = utils.grad_r(phi[j], dx, dim)
            ccphix = 0.5*(gphi[0]+np.roll(gphi[0], 1, 0))
            ccphiy = 0.5*(gphi[1]+np.roll(gphi[1], 1, 1))
            smgphi2 += (ccphix**2 + ccphiy**2)
            vasmgphi2 += 0.25*((gphi[0]+np.roll(gphi[0], -1, 1))**2)
            vasmgphi2 += 0.25*((gphi[1]+np.roll(gphi[1], -1, 0))**2)
            dTe2dphi.append(T*va_e2*utils.grad2(phi[j], dx, dim) + va_e2*(gT[0]*ccphix + gT[1]*ccphiy) + T*(e2_x*ccphix + e2_y*ccphiy))
            #phi term used for computing mobilities
            phi_m = 0.5*(phi[j] + np.roll(phi[j], 1, 0))
            phi_m = 0.5*(phi_m + np.roll(phi_m, 1, 1))
            phi_m = 0.5*(phi_m + np.roll(phi_m, -1, 0))
            phi_m = 0.5*(phi_m + np.roll(phi_m, -1, 1))
            with np.errstate(divide='ignore', invalid='ignore'):
                #this should be absolute valued, but since we use a cutoff function, there are no negative values
                pbp.append(phi_m/(1-phi_m))
            
        #kij matrix computation
        for j in range(len(phi)):
            kij.append([])
            for k in range(j):
                with np.errstate(divide='ignore', invalid='ignore'):
                    kij_value = kijbar[j][k]*pbp[j]*pbp[k]
                kij_value[np.isnan(kij_value)] = 0 #(values that come up as nan only happens if one order param equals 1, so no phi motion can happen anyways)
                kij_value[np.isinf(kij_value)] = 0 #(same for +/- inf)
                kij[j].append(kij_value)
                kij[k].append(kij_value)
            kij[j].append(0)
            
        #kij matrix computation round 2: set diagonal elements to negative sum of row
        for j in range(len(phi)):
            diag_element = 0
            for k in range(len(phi)):
                diag_element += kij[j][k]
            kij[j][j] -= diag_element
        
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
        smallest = 5000.
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
            noise_c=np.random.normal(0, std_c, q1.shape)
            temp = 0 #mobility
            temp2 = 0 #energy
            for k in range(len(phi)):
                temp += h[k]*D[k]
                temp2 += h[k]*dGdc[k][j]
            M_c.append(v_m*c[j]*temp/R/T)
            #add the change in noise inside the functional
            dFdc.append(temp2/v_m+noise_c)
        for j in range(len(c)):
            deltac.append(utils.divagradb(M_c[j]*(1-c[j]), dFdc[j], dx, dim))
            for k in range(len(c)):
                if not (j == k):
                    deltac[j] -= utils.divagradb(M_c[j]*c[k], dFdc[k], dx, dim)
        
        dFdphi = []
        for j in range(len(phi)):
            #noise in phi, based on Langevin Noise
            std_phi=np.sqrt(np.absolute(2*R*T/v_m/np.max(kijbar)))
            noise_phi=np.random.normal(0, std_phi, phi[j].shape)
            
            #compute dFdphi terms, including the langevin noise
            dFdphi.append(0.5*T*de2dpj[j]*smgphi2 + hp[j]*G[j]/v_m+dwdpj[j]*g*T + w*gp[j]*T+H[j]*T*(2*phi[j])*rgqs_0 - utils.vg_ul(0.5*va_T*de2dpxj[j]*vasmgphi2, dim, 0, dx) - utils.vg_ul(0.5*va_T*de2dpyj[j]*vasmgphi2, dim, 1, dx) - dTe2dphi[j])#+noise_phi
        
        deltaphi = []
        for k in range(len(phi)):
            deltaphi.append(0)
            for j in range(len(phi)):
                deltaphi[k] += kij[k][j]*dFdphi[j]
        
        #changes in q, part 1
        dq_component = 0
        for j in range(len(phi)):
            dq_component += p[j]*H[j]
        dq_component *= T
    
        gaq1 = utils.gaq(gq1l, gq1r, rgqsl, rgqsr, dq_component, dx, dim)
        gaq4 = utils.gaq(gq4l, gq4r, rgqsl, rgqsr, dq_component, dx, dim)
        
        lq1 = utils.grad2(q1, dx, dim)
        lq4 = utils.grad2(q4, dx, dim)
        t1 = eqbar*eqbar*lq1+(gaq1) + 0.5*T*de2dq1*smgphi2
        t4 = eqbar*eqbar*lq4+(gaq4) + 0.5*T*de2dq4*smgphi2
        lmbda = (q1*t1+q4*t4)
        deltaq1 = M_q*(t1-q1*lmbda)
        deltaq4 = M_q*(t4-q4*lmbda)
    
        #check for questionable points if DEBUG
        if(DEBUG):
            weirdpoints = np.argwhere(deltaphi[0] > 0.5/dt)
            for j in range(1, len(phi)):
                weirdpoints = np.concatenate((weirdpoints, np.argwhere(deltaphi[j] > 0.5/dt)), axis=0)
            for j in range(len(phi)):
                weirdpoints = np.concatenate((weirdpoints, np.argwhere(phi[j] > 1.01)), axis=0)
            for j in range(len(phi)):
                weirdpoints = np.concatenate((weirdpoints, np.argwhere(phi[j] < -0.01)), axis=0)
            if(len(weirdpoints) > 0):
                weirdpoints = np.unique(weirdpoints, axis=0)
                print("Weird points: ", weirdpoints)
                weirdpoints = tuple(weirdpoints.T)
                for j in range(len(phi)):
                    print("phi["+str(j)+"]: ", phi[j][weirdpoints])
                for j in range(len(phi)):
                    for k in range(len(phi)):
                        print("kij["+str(j)+"]["+str(k)+"]: ", kij[j][k][weirdpoints])
                for j in range(len(c)):
                    print("c["+str(j)+"]: ", c[j][weirdpoints])
                for j in range(len(phi)):
                    print("deltaphi["+str(j)+"]: ", deltaphi[j][weirdpoints])
                for j in range(len(phi)):
                    print("0.5*T*de2dpj["+str(j)+"]*smgphi2: ", (0.5*T*de2dpj[0]*smgphi2)[weirdpoints])
                for j in range(len(phi)):
                    print("hp["+str(j)+"]*G["+str(j)+"]/v_m: ", (hp[j]*G[j]/v_m)[weirdpoints])
                for j in range(len(phi)):
                    print("dwdpj["+str(j)+"]*g*T: ", (dwdpj[j]*g*T)[weirdpoints])
                for j in range(len(phi)):
                    print("w*gp["+str(j)+"]*T: ", (w*gp[j]*T)[weirdpoints])
                for j in range(len(phi)):
                    print("H["+str(j)+"]*T*(2*phi["+str(j)+"])*rgqs_0: ", (H[j]*T*(2*phi[j])*rgqs_0)[weirdpoints])
                for j in range(len(phi)):
                    print("-utils.vg_ul(0.5*va_T*de2dpxj["+str(j)+"]*vasmgphi2, dim, 0, dx): ", (-utils.vg_ul(0.5*va_T*de2dpxj[j]*vasmgphi2, dim, 0, dx))[weirdpoints])
                for j in range(len(phi)):
                    print("-utils.vg_ul(0.5*va_T*de2dpyj["+str(j)+"]*vasmgphi2, dim, 1, dx): ", (-utils.vg_ul(0.5*va_T*de2dpyj[j]*vasmgphi2, dim, 1, dx))[weirdpoints])
                for j in range(len(phi)):
                    print("-dTe2dphi["+str(j)+"]: ", (-dTe2dphi[j])[weirdpoints])
    
        #apply changes
        for j in range(len(c)):
            #print(c)
            #print(deltac)
            c[j] += deltac[j]*dt
        
        for j in range(len(phi)):
            phi[j] += deltaphi[j]*dt
            
        #apply multiObstacle function to phi, clipping values outside 0 to 1
        phi = utils.multiObstacle(phi)
        
        q1 += deltaq1*dt
        q4 += deltaq4*dt
        
        #print("phi", phi[0][0])
        utils.applyBCs_ncnp(phi, c, q1, q4, nbc)
        T += dTdt
        if(i%10 == 0):
            q1, q4 = utils.renormalize(q1, q4)
        
        #This code segment prints the progress after every 5% of the simulation is done (for convenience)
        if(steps > 19):
            if(i%(steps/20) == 0):
                print(str(5*i/(steps/20))+"% done...")
    
        #This code segment saves the arrays every 500 steps, and adds nuclei
        #note: nuclei are added *before* saving the data, so stray nuclei may be found before evolving the system
        #***NUCLEATION TEMPORARILY TURNED OFF FOR MULTIORDER TESTING***
        if(step%500 == 0):
            #find the stochastic nucleation critical probabilistic cutoff
            #attn -- Q and T_liq are hard coded parameters for Ni-10%Cu
            #Q0=8*10**5 #activation energy of migration
            #T_liq=1697 #Temperature of Liquidus (K)
            #J0,p11=utils.find_Pn(T_liq, T, Q0, dt) 
            #print(J0)
            #phi, q1, q4=utils.add_nuclei(phi, q1, q4, p11, len(phi))
            utils.saveArrays_ncnp(data_path, step, phi, c, q1, q4)
        if(i == steps-1):
            utils.saveArrays_ncnp(data_path, step, phi, c, q1, q4)
        if(DEBUG):
            print("Maximum value of w", np.max(w))
            print("Maximum value of G[0]", np.max(G[0]))
            print("Maximum value of phi noise", np.max(noise_phi))
            print("Maximum value of 0.5*T*de2dpj[0]*smgphi2", np.max(np.absolute(0.5*T*de2dpj[j]*smgphi2)))
            print("Maximum value of h'(p_0)*G/v_m", np.max(np.absolute(hp[0]*G[0]/v_m)))
            print("Maximum value of dw/dp_0 *g*T", np.max(np.absolute(dwdpj[0]*g*T)))
            print("Maximum value of w*dg/dp_0 *T", np.max(np.absolute(w*gp[0]*T)))
            print("Maximum value of H(p_0)*T*p/mgq", np.max(np.absolute(H[0]*T*(2*phi[0])*rgqs_0)))
            print("Maximum value of del x component of dFdphi", np.max(np.absolute(-utils.vg_ul(0.5*va_T*de2dpxj[0]*vasmgphi2, dim, 0, dx))))
            print("Maximum value of del y component of dFdphi", np.max(np.absolute(-utils.vg_ul(0.5*va_T*de2dpyj[0]*vasmgphi2, dim, 1, dx))))
            print("Maximum value of dTe2dphi[0]", np.max(np.absolute(-dTe2dphi[0])))
            print("Maximum value of deltaphi[0]", np.max(np.absolute(deltaphi[0])))
            if(len(phi) == 2):
                print("Sum of phi terms (should be one)", np.max(phi[0]+phi[1]))
            if(len(phi) == 3):
                print("Sum of phi terms (should be one)", np.max(phi[0]+phi[1]+phi[2]))

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
        init_tdb_vars_ncnp(utils.tdb)
        simulate_ncnp(data_path, initial_step, steps, init_T, grad_T, dTdt)
    else:
        print("Error! Needs exactly 6 additional arguments! (data_path, initial_step, steps, init_T, grad_T, dT/dt)")