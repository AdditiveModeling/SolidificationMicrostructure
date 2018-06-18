
# coding: utf-8

# # Anisotropic Dorr Model
# 
# This model begins with the following Free Energy functional, with terms taken from Pusztai2005 and Dorr2010:
# 
# $$ F = \int_V (\frac{\epsilon_\phi^2}{2}|\nabla \phi|^2 + f(\phi,c) + f_{ori} + \frac{\epsilon_q^2}{2}|\nabla \textbf{q}|^2) dV \qquad (1)$$ 
# 
# In order, these terms are:
# 
# * The interfacial energy in phi
# * The bulk free energy
# * The orientational energy, representing the energy of lattice mismatch
# * The interfacial energy in the orientation (q), artificially added to have smooth orientation transitions
# 
# This expression is identical to that presented in Dorr2010. To add anisotropic growth, we add a directional component ($\eta$), to the interfacial energy in phi:
# 
# $$ F = \int_V (\frac{\epsilon_\phi^2}{2} \eta|\nabla \phi|^2 + f(\phi,c) + f_{ori} + \frac{\epsilon_q^2}{2}|\nabla \textbf{q}|^2) dV \qquad (1)$$ 
# 
# We use the following expression to promote cubic symmetry in the PFM:
# 
# $$ \eta = 1 - 3\gamma_\epsilon + 4\gamma_\epsilon\frac{\psi_x^4 + \psi_y^4 + \psi_z^4}{|\nabla \phi|^4} $$ 
# 
# Here, $\gamma_\epsilon$ represents the magnitude of the anisotropy in the interfacial energy, and all $\psi$ terms are found from rotating the components of $\nabla \phi$ by $\textbf{q}$:
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = (q_1 + q_2\textbf{i} + q_3\textbf{j} + q_4\textbf{k})*(\phi_x\textbf{i} + \phi_y\textbf{j} + \phi_z\textbf{k})*(q_1 - q_2\textbf{i} - q_3\textbf{j} - q_4\textbf{k})$$
# 
# It is important to note that, since we are using quaternions, certain multiplications are non-commutative.
# 
# $$\textbf{i}\textbf{i} = \textbf{j}\textbf{j} = \textbf{k}\textbf{k} = -1, \textbf{i}\textbf{j} = \textbf{k}, \textbf{j}\textbf{k} = \textbf{i}, \textbf{k}\textbf{i} = \textbf{j}, \textbf{j}\textbf{i} = \textbf{-k}, \textbf{k}\textbf{j} = \textbf{-i}, \textbf{i}\textbf{k} = \textbf{-j}$$
# 
# In 2D, the only meaningful rotation is within the xy plane, so we require that $q_2 = q_3 = 0$ for the 2D model. Additionally, $\phi_z = 0$. As a consequence, the expressions for $\psi$ simplify in the following manner:
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = (q_1 + q_4\textbf{k})*(\phi_x\textbf{i} + \phi_y\textbf{j})*(q_1 - q_4\textbf{k})$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = (q_1\phi_x\textbf{i} + q_1\phi_y\textbf{j} + q_4\phi_x\textbf{k}\textbf{i} + q_4\phi_y\textbf{k}\textbf{j})*(q_1 - q_4\textbf{k})$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = (q_1\phi_x\textbf{i} + q_1\phi_y\textbf{j} + q_4\phi_x\textbf{j} - q_4\phi_y\textbf{i})*(q_1 - q_4\textbf{k})$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = q_1^2\phi_x\textbf{i} + q_1^2\phi_y\textbf{j} + q_1q_4\phi_x\textbf{j} - q_1q_4\phi_y\textbf{i} - q_1q_4\phi_x\textbf{i}\textbf{k} - q_1q_4\phi_y\textbf{j}\textbf{k} - q_4^2\phi_x\textbf{j}\textbf{k} + q_4^2\phi_y\textbf{i}\textbf{k}$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = q_1^2\phi_x\textbf{i} + q_1^2\phi_y\textbf{j} + q_1q_4\phi_x\textbf{j} - q_1q_4\phi_y\textbf{i} + q_1q_4\phi_x\textbf{j} - q_1q_4\phi_y\textbf{i} - q_4^2\phi_x\textbf{i} - q_4^2\phi_y\textbf{j}$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = q_1^2\phi_x\textbf{i} - q_1q_4\phi_y\textbf{i} - q_1q_4\phi_y\textbf{i} - q_4^2\phi_x\textbf{i} + q_1^2\phi_y\textbf{j} + q_1q_4\phi_x\textbf{j} + q_1q_4\phi_x\textbf{j}  - q_4^2\phi_y\textbf{j}$$
# 
# $$\psi_x\textbf{i} + \psi_y\textbf{j} + \psi_z\textbf{k} = ((q_1^2 - q_4^2)\phi_x - 2q_1q_4\phi_y)\textbf{i} + ((q_1^2 - q_4^2)\phi_y + 2q_1q_4)\textbf{j}$$
# 
# Therefore, in 2D, $\psi_x = (q_1^2 - q_4^2)\phi_x - 2q_1q_4\phi_y$, and $\psi_y = (q_1^2 - q_4^2)\phi_y + 2q_1q_4$. Additionally, $\psi_z = 0$, as expected for a 2D model.
# 
# In this equation, the bulk free energy, and the orientational mismatch energy are represented by the following equations:
# 
# $$ f(\phi, c) = Wg(\phi) + p(\phi)f_S + (1-p(\phi)f_L \qquad (2)$$
# 
# $$ 

# In[3]:

import numpy as np
import os

# In[62]:

def __h(phi):
    return phi*phi*phi*(10-15*phi+6*phi*phi)

def __hprime(phi):
    return (30*phi*phi*(1-phi)*(1-phi))

def __g(phi):
    return (phi*phi*(1-phi)*(1-phi))

def __gprime(phi):
    return (4*phi*phi*phi - 6*phi*phi +2*phi)

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

def grad_l(phi, dx, i):
    r = []
    for i in range(dim):
        phim = np.roll(phi, 1, i)
        r.append((phi-phim)/(dx))
    return r

def grad_r(phi, dx, i):
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
    return timestep, _c, _phi, _q1, _q4

def saveArrays(path, timestep, c, phi, q1, q4):
    np.save(path+'phi_'+str(timestep), phi)
    np.save(path+'c_'+str(timestep), c)
    np.save(path+'q1_'+str(timestep), q1)
    np.save(path+'q4_'+str(timestep), q4)


# In[136]:

#temperature
T = 1574.

#bcc = 0L = e, fcc = 1S = d
#material parameters, J, cm, K, s (except for R and Q terms, which use joules)
M_qmax = 80000000. #maximum mobility of orientation, 1/(s*J)
H = 1e-11 #interfacial energy term for quaternions, J/(K*cm)

#material parameters, from Warren1995
T_mA = 1728. #melting point of nickel
T_mB = 1358. #melting point of copper
L_A = 2350. #latent heat of nickel, J/cm^3
L_B = 1728. #latent heat of copper, J/cm^3
s_A = 0.000037 #surface energy of nickel, J/cm^2
s_B = 0.000029 #surface energy of copper, J/cm^2
D_L = 1e-5 #diffusion in liquid, cm^2/s
D_S = 1e-9 #diffusion in solid, cm^2/s
B_A = 0.33 #linear kinetic coefficient of nickel, cm/K/s
B_B = 0.39 #linear kinetic coefficient of copper, cm/K/s
v_m = 7.42 #molar volume, cm^3/mol
R = 8.314 #gas constant, J/mol*K
y_e = 0.12 #anisotropy

#discretization params
dx = 4.6e-6 #spacial division, cm
dt = dx*dx/5./D_L/8
d = dx/0.94 #interfacial thickness

#discretization dependent params, since d is constant, we compute them here for now
ebar = np.sqrt(6*np.sqrt(2)*s_A*d/T_mA) #baseline energy
eqbar = 0.5*ebar
W_A = 3*s_A/(np.sqrt(2)*T_mA*d)
W_B = 3*s_B/(np.sqrt(2)*T_mB*d)
M_A = T_mA*T_mA*B_A/(6*np.sqrt(2)*L_A*d)
M_B = T_mB*T_mB*B_B/(6*np.sqrt(2)*L_B*d)

#tests
print('dt = '+str(dt))
print('ebar = '+str(ebar))


# In[187]:

#this code block initializes the simulation

np.set_printoptions(threshold=np.inf)

shape = []
dim = 2
res = 2000
seeds = 40
for i in range(dim):
    shape.append(res)

c = np.zeros(shape)
phi = np.zeros(shape)
q1 = np.zeros(shape)
q4 = np.zeros(shape)
q1 += np.cos(1*np.pi/8)
q4 += np.sin(1*np.pi/8)
c += 0.40831

randAngle = np.random.rand(seeds)*np.pi/4
randX = np.random.rand(seeds)*(res-8)+4
randY = np.random.rand(seeds)*(res-8)+4
for k in range(seeds):
    for i in range((int)(randX[k]-6), (int)(randX[k]+6)):
        for j in range((int)(randY[k]-6), (int)(randY[k]+6)):
            if((i-randX[k])*(i-randX[k])+(j-randY[k])*(j-randY[k]) < 16):
                phi[i][j] = 1
                q1[i][j] = np.cos(randAngle[k])
                q4[i][j] = np.sin(randAngle[k])

reference = np.zeros(res)
for i in range(res):
    reference[i] = i

step = 0


# In[189]:

# This block runs the simulation for some number of time steps

#add these values after evolving the simulation to almost stability - to test if the interface in phi is truly stable
#phi = np.zeros(res)
#phi += 1

time_steps = 16000

# original number is the one used in Pusztai2005 and Granasy2014
# multiplication factor is used to ensure q is stable

for i in range(time_steps):
    step += 1
    #print(i)
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
        pp = 2*phi
        hprime = _hprime(phi)
        gprime = _gprime(phi)
    
        #bulk energy terms, using ideal solution model from Warren1995
        H_A = W_A*gprime - 30*L_A*(1/T-1/T_mA)*g
        H_B = W_B*gprime - 30*L_B*(1/T-1/T_mB)*g
    
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
        
        a2_b2 = vertex_averaged_q1*vertex_averaged_q1-vertex_averaged_q4*vertex_averaged_q4
        ab2 = 2.*vertex_averaged_q1*vertex_averaged_q4
        
        vertex_centered_gpsi = []
        vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[0] - ab2*vertex_averaged_gphi[1])
        vertex_centered_gpsi.append(a2_b2*vertex_averaged_gphi[1] + ab2*vertex_averaged_gphi[0])
        
        psi_xxy = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[1]
        psi_xyy = vertex_centered_gpsi[0]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
        psi_xxyy = psi_xxy*vertex_centered_gpsi[1]
        #psi_x4y4 = vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]*vertex_centered_gpsi[0]
        #psi_x4y4 += vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]*vertex_centered_gpsi[1]
        
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
        
        #vertex_centered_eta = 1 - 3*y_e + 4*y_e*psi_x4y4/(vertex_centered_mgphi2*vertex_centered_mgphi2)
        vertex_centered_eta = 1. +y_e -8.*y_e*psi_xxyy/(vertex_centered_mgphi2*vertex_centered_mgphi2)
        #vertex_centered_eta = np.ones_like(q1)
        cell_centered_eta = (vertex_centered_eta + np.roll(vertex_centered_eta, -1, 0))/2.
        cell_centered_eta = (cell_centered_eta + np.roll(cell_centered_eta, -1, 1))/2.
        
        phi_temp_x = (a2_b2*psi_xyy + ab2*psi_xxy - (2*vertex_averaged_gphi[0]*psi_xxyy)/vertex_centered_mgphi2)/vertex_centered_mgphi2
        phi_temp_y = (a2_b2*psi_xxy - ab2*psi_xyy - (2*vertex_averaged_gphi[1]*psi_xxyy)/vertex_centered_mgphi2)/vertex_centered_mgphi2
        dptxdx = partial_r(phi_temp_x, dx, 0)
        cell_centered_dptxdx = (dptxdx + np.roll(dptxdx, -1, 1))/2.
        dptydy = partial_r(phi_temp_y, dx, 1)
        cell_centered_dptydy = (dptydy + np.roll(dptydy, -1, 0))/2.
        
        #change in c
        D_C = D_S+m*(D_L-D_S)
        temp = D_C*v_m*c*(1-c)*(H_B-H_A)/R
        deltac = divagradb(D_C, c, dx, dim) + divagradb(temp, phi, dx, dim)
        
        #change in phi
        lphi = grad2(phi, dx, 2)
        g_eta_g_phi_x = partial_r(vertex_centered_eta, dx, 0)
        g_eta_g_phi_x = 0.25*(gphi[0]+np.roll(gphi[0], -1, 0))*(g_eta_g_phi_x+np.roll(g_eta_g_phi_x, -1, 1))
        g_eta_g_phi_y = partial_r(vertex_centered_eta, dx, 1)
        g_eta_g_phi_y = 0.25*(gphi[1]+np.roll(gphi[1], -1, 1))*(g_eta_g_phi_y+np.roll(g_eta_g_phi_y, -1, 0))
        g_eta_g_phi = g_eta_g_phi_x + g_eta_g_phi_y
        M_phi = (1-c)*M_A + c*M_B
        deltaphi = M_phi*(ebar*ebar*(cell_centered_eta*lphi + g_eta_g_phi -8*y_e*cell_centered_dptxdx -8*y_e*cell_centered_dptydy)-(1-c)*H_A-c*H_B-2*H*T*pp*rgqs_0)
        randArray = 2*np.random.rand(res, res)-1
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
        
        t1_temp = (16*ebar*ebar*y_e/vertex_centered_mgphi2)*(psi_xyy*(q1px - q4py) + psi_xxy*(q1py + q4px))
        t4_temp = (16*ebar*ebar*y_e/vertex_centered_mgphi2)*(psi_xyy*(-q4px - q1py) + psi_xxy*(-q4py + q1px))
        cc_t1_temp = (t1_temp + np.roll(t1_temp, -1, 0))/2.
        cc_t1_temp = (cc_t1_temp + np.roll(cc_t1_temp, -1, 1))/2.
        cc_t4_temp = (t4_temp + np.roll(t4_temp, -1, 0))/2.
        cc_t4_temp = (cc_t4_temp + np.roll(cc_t4_temp, -1, 1))/2.
        
        t1 = eqbar*eqbar*lq1+(gaq1) + cc_t1_temp
        t4 = eqbar*eqbar*lq4+(gaq4) + cc_t4_temp
        lmbda = (q1*t1+q4*t4)
        deltaq1 = M_q*(t1-q1*lmbda)
        deltaq4 = M_q*(t4-q4*lmbda)
    
    #changes in q
    
    
    
    #apply changes
    c += deltac*dt
    phi += deltaphi*dt
    q1 += deltaq1*dt
    q4 += deltaq4*dt
    if(i%10 == 0):
        q1, q4 = renormalize(q1, q4)
        
    #This code segment prints the progress after every 5% of the simulation is done (for convenience)
    if(i%(time_steps/20) == 0):
        print(str(5*i/(time_steps/20))+"% done...")
    
    #This code segment saves the arrays every 400 steps
    if(step%400 == 0):
        saveArrays("data/", step, c, phi, q1, q2, q3, q4)

print("Done")