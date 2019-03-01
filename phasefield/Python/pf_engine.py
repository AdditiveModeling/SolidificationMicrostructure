import numpy as np
import sys 
import os
import pf_utils as utils

#dimension (only 2D for python at the moment!)
dim = 2

#material parameters, J, cm, K, s
M_qmax = 80000000./1574. #maximum mobility of orientation, 1/(s*J)
H = 1e-11 #interfacial energy term for quaternions, J/(K*cm)

#material parameters, mostly from Warren1995, some exceptions (anisotropy, e_SX, C_X)
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

def simulate(path, nbc, initialStep, steps, gradT, dTdt):
    if not os.path.isfile(path+"info.txt"):
        print("Simulation has not been initialized yet - aborting engine!")
        return
    
    info = open(path+"info.txt", 'a')
    info.write("  New Simulation Run:\n")
    info.write("    Initial Step: "+str(initialStep)+"\n")
    info.write("    Number of steps to run: "+str(steps)+"\n")
    info.write("    Neumann Boundary Condition array used: ["+str(nbc[0])+", "+str(nbc[1])+"]\n")
    info.write("    Temperature gradient (K/cell): "+str(gradT)+"\n")
    info.write("    Change in temperature over time (K/time_step): "+str(dTdt)+"\n\n")
    info.close()
    
    step, phi, c, q1, q4 = utils.loadArrays(path, initialStep)
    shape = c.shape #get the shape of the simulation region from one of the arrays
    
    #temperature
    T = 1574.*np.ones(shape)
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
            hprime = utils._hprime(phi)
            gprime = utils._gprime(phi)
    
            #bulk energy terms, using ideal solution model from Warren1995
            H_A = W_A*gprime*T - 30*L_A*(1-T/T_mA)*g
            H_B = W_B*gprime*T - 30*L_B*(1-T/T_mB)*g
    
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
            G_L, dGLdc = utils.compute_tdb_energy(T, c, "LIQUID")
            G_S, dGSdc = utils.compute_tdb_energy(T, c, "FCC_A1")
        
            #change in c
            M_C = v_m*c*(1-c)*(D_S+m*(D_L-D_S))/R/1574.
            temp = (dGSdc + m*(dGLdc-dGSdc))/v_m + (W_B-W_A)*g*T
            deltac = utils.divagradb(M_C, temp, dx, dim)
        
            #change in phi
            divTgradphi = utils.divagradb(T, phi, dx, dim)
            M_phi = (1-c)*M_A + c*M_B
        
            psix3 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
            psiy3 = vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
            pf_comp_x = 8*y_e*T*((2*a2_b2*psix3 + 2*ab2*psiy3)/vertex_centered_mgphi2 - vertex_averaged_gphi[0]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
            pf_comp_x = (np.roll(pf_comp_x, -1, 0) - pf_comp_x)/dx
            pf_comp_x = (np.roll(pf_comp_x, -1, 1) + pf_comp_x)/2.
            pf_comp_y = 8*y_e*T*((2*a2_b2*psiy3 - 2*ab2*psix3)/vertex_centered_mgphi2 - vertex_averaged_gphi[1]*(psix3*vertex_centered_gpsi[0] + psiy3*vertex_centered_gpsi[1])/(vertex_centered_mgphi2*vertex_centered_mgphi2))
            pf_comp_y = (np.roll(pf_comp_y, -1, 1) - pf_comp_y)/dx
            pf_comp_y = (np.roll(pf_comp_y, -1, 0) + pf_comp_y)/2.
            deltaphi = M_phi*(ebar*ebar*((1-3*y_e)*divTgradphi + pf_comp_x + pf_comp_y)-30*g*(G_S-G_L)/v_m-W_A*gprime*T*(1-c)-W_B*gprime*T*c-4*H*T*phi*rgqs_0*1574.)
            randArray = 2*np.random.random_sample(shape)-1
            alpha = 0.3
            deltaphi += M_phi*alpha*randArray*(16*g)*(30*g*(G_S-G_L)/v_m+W_A*T*gprime*(1-c)+W_B*T*gprime*c)
        
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
        c += deltac*dt
        phi += deltaphi*dt
        q1 += deltaq1*dt
        q4 += deltaq4*dt
        utils.applyBCs(c, phi, q1, q4, nbc)
        T += dTdt
        if(i%10 == 0):
            q1, q4 = utils.renormalize(q1, q4)
        
        #This code segment prints the progress after every 5% of the simulation is done (for convenience)
        if(steps > 19):
            if(i%(steps/20) == 0):
                print(str(5*i/(steps/20))+"% done...")
    
        #This code segment saves the arrays every 1000 steps
        if(step%500 == 0):
            utils.saveArrays(path, step, phi, c, q1, q4)
            print((-30*g*(G_S-G_L)/v_m)[40])
            print(((1-c)*(-30*L_A*(1-T/T_mA)*g)+c*(-30*L_B*(1-T/T_mB)*g))[40])

    print("Done")

#path, _nbc_x, _nbc_y, initialStep, steps, these are the command line arguments

if __name__ == '__main__':
    if(len(sys.argv) == 8):
        path = sys.argv[1]
        if not os.path.exists(path):
            os.makedirs(path)
    
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
        
        initialStep = int(sys.argv[4])
        steps = int(sys.argv[5])
        gradT = float(sys.argv[6])
        dTdt = float(sys.argv[7])
        simulate(path, nbc, initialStep, steps, gradT, dTdt)
    else:
        print("Error! Needs exactly 7 additional arguments! (path, nbc_x, nbc_y, initialStep, steps, gradT, dT/dt)")