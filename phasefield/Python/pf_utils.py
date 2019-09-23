import numpy as np
import sympy as sp
import re
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pycalphad as pyc

#initialize TDB file in pycalphad, from up a dir and in the TDB folder
root_folder = os.path.abspath(os.path.dirname(__file__) + '/..')
tdb_path = 'Ni-Cu.tdb'
tdb = pyc.Database(root_folder + '/TDB/Ni-Cu.tdb')
phases = ['LIQUID', 'FCC_A1']
components = ['CU', 'NI']

def find_Pn(T_M, T, Q ,dt):
    #finding the probability of forming a critical nucleus, nucleating only every 500 time steps
    #input: T_M -- temperature of the liquidus, T -- Temperature, Q -- activation energy for migration
    #choose free parameters a and b
    #code by Vera Titze
    a=10**28
    b=2.8*10**3
    e=2.7182818284590452353
    R=8.314
    J0=a*e**(-b/(T_M-T))*e**(-Q/(R*T))
    Pn=1-e**(-J0*dt*500)
    return J0,Pn

def add_nuclei(phi, q1, q4, p11, size):
    #adds nuclei to the phi, q1, and q4 fields for a given probability array, p11
    #code by Vera Titze
    random=np.random.random((size, size))
    nuclei_centers=np.argwhere(random<p11)
    print('number of nuclei added: ',len(nuclei_centers))
    for center in nuclei_centers:
        angle=np.random.random()
        for i in range((int)(center[0]-5), (int)(center[0]+5)):
            for j in range((int)(center[1]-5), (int)(center[1]+5)):
                if (i>=0 and i<size and j<size and j>=0):
                    if((i-center[0])*(i-center[0])+(j-center[1])*(j-center[1]) < 25):
                        if(phi[i][j]<0.2):
                            phi[i][j] = 1
                            q1[i][j] = np.cos(angle*2*np.pi)
                            q4[i][j] = np.sin(angle*2*np.pi)
    return phi, q1, q4

def compute_tdb_energy_nc(temps, c, phase):
    """
    Computes Gibbs Free Energy and its derivative*S* w.r.t. composition, for a given temperature field and list of composition fields
    Derivatives are computed by holding all other explicit composition variables constant
    c_i is increased, c_N is decreased (the implicitly-defined last composition variable which equals 1-sum(c_i) )
    
    Returns GM (Molar Gibbs Free Energy) and dGdci (list of derivatives of GM, w.r.t. c_i)
    """
    #alphabetical order of components!
    fec = [] #flattened expanded c
    for i in range(len(c)):
            fec.append(np.expand_dims(c[i].flatten(), axis=1))
    fec_n_comp = np.ones(fec[0].shape)
    for i in range(len(c)):
        fec_n_comp -= fec[i]
    for i in range(len(c)):
        fec_n_comp = np.concatenate((fec_n_comp, fec[i]), axis=1)
    #move final component to end, maybe ill find a way to write this better in the future...
    fec_n_comp = np.roll(fec_n_comp, -1, axis=1)
    #offset composition, for computing slope of GM w.r.t. comp
    fec_nc_offset = []
    for i in range(len(c)):
        fec_offset = np.zeros([len(c)+1])
        fec_offset[i] = 0.0000001
        fec_offset[len(c)] = -0.0000001
        fec_nc_offset.append(fec_n_comp+fec_offset)
    flattened_t = temps.flatten()
    GM = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=fec_n_comp, broadcast=False).GM.values.reshape(c[0].shape)
    GM_derivs = []
    for i in range(len(c)):
        GM_derivs.append((pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=fec_nc_offset[i], broadcast=False).GM.values.reshape(c[0].shape)-GM)*(10000000.))
    return GM, GM_derivs

def load_tdb(tdb_path):
    """
    loads the TDB file at the specified path, and updates global variables accordingly
    """
    global tdb
    global phases
    global components
    if not os.path.isfile(root_folder+"/TDB/"+tdb_path):
        print("utils.load_tdb Error: TDB file does not exist!")
        return False
    tdb = pyc.Database(root_folder + '/TDB/' + tdb_path)
    
    #update phases
    phases = list(tdb.phases)
    phases.sort()
    
    #update components
    components = list(tdb.elements)
    components.sort()
    return True

def __h(phi_i):
    #h function from Dorr2010
    #important, input is a single field of the phi array
    phi2 = phi_i**2
    return (phi2*phi_i)*(10-15*phi_i+6*phi2)

def __hprime(phi_i):
    #derivative of the h function from Dorr2010, w.r.t phi
    #important, input is a single field of the phi array
    phi_phi2 = phi_i-(phi_i**2)
    return 30*(phi_phi2**2)

#Numpy vectorized versions of above functions
_h = np.vectorize(__h)
_hprime = np.vectorize(__hprime)

def _g(phi):
    #g function from Toth2015, for multiorder simulations
    #important: input is an array of phase fields, not a single field!
    sum = 1/12.
    for i in range(len(phi)):
        phi2 = phi[i]**2
        sum += phi2*phi2/4. - phi2*phi[i]/3.
        for j in range(i+1, len(phi)):
            sum += phi2*phi[j]**2
    return sum

def _gprime(phi, index):
    #derivative of g function, w.r.t phi_index
    sum = 0;
    for i in range(len(phi)):
        sum += phi[i]**2
    phi2 = phi[index]**2
    sum -= phi2
    return phi2*phi[index] - phi2 + phi[index]*sum
    

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

def va_br(field, dim):
    #vertex averaged value of field, at the bottom-right vertex
    f = 0.5*(field+np.roll(field, -1, 0))
    for i in range(1,dim):
        f = 0.5*(f+np.roll(f,-1,i))
    return f

def va_ul(field, dim):
    #vertex averaged value of field, at the upper-left vertex
    f = 0.5*(field+np.roll(field, 1, 0))
    for i in range(1,dim):
        f = 0.5*(f+np.roll(f,1,i))
    return f

def vg_ul(field, dim, direction, dx):
    f = field
    for i in range(dim):
        if(i != direction):
            f = f + np.roll(f, 1, i)
    f = (f-np.roll(f, 1, direction))/dx
    return f

def doublesums(phi, q1, q4, ebar2, _w, gamma, dx):
    #utility function which computes the double summations present in the rate equations
    #outputs values which are close to the final calculations:
    #         e2, w, de2dpj, de2dpxj, de2dpyj, dwdpj, de2dq1, de2dq4
    #e2: Vertex-centered value for e2, used in div dot dI/d-grad-phi_j computations
    #w: cell-centered value for w, used in dI/dphi_j
    #de2dpj: cell centered value of de2/dphi_j, used in dI/dphi_j
    #de2dpxj/de2dpyj: vertex-centered values for de2/dphi_{x/y, j}
    #dwdpj: cell centered value of dw/dphi_j, used in dI/dphi_j
    #de2dq1/de2dq4: vertex averaged, cell centered value for de2/dq_{1,4}
    
    dim = 2 #HARD CODED, MUST BE CHANGED FOR 3D!
    
    zeros = np.zeros(q1.shape)
    phi2 = [] #vector of phi_i^2 values
    vaphi = [] #vector of vertex averaged phi_i values
    vaphi2 = [] #vector of vertex averaged phi_i^2 values
    vaphi4 = [] #vector of vertex averaged phi_i^4 values
    phi3 = [] #vector of phi_i^3 values
    phi4 = [] #vector of phi_i^4 values
    eta_ij = [] #matrix of eta values
    psix_ij = [] #matrix of psi_x,ij values, br_vertex
    psiy_ij = [] #matrix of psi_y,ij values, br_vertex
    phix = [] #vector of phi_x values, r_face
    phiy = [] #vector of phi_y values, b_face
    phix_ij = [] #matrix of phix_i - phix_j values, br_vertex
    phiy_ij = [] #matrix of phiy_i - phiy_j values, br_vertex
    #vertex averaged quaternion values, for psi computations
    #with np.errstate(divide='ignore', invalid='ignore'):
    with np.errstate(divide='warn', invalid='warn'):
        va_q1 = va_br(q1, dim)
        va_q4 = va_br(q4, dim)
        q2q2 = va_q1**2 - va_q4**2
        qq2 = 2*va_q1*va_q4
        for i in range(len(phi)):
            phi2.append(phi[i]**2)
            phi4.append(phi2[i]**2)
            phi3.append(phi2[i]*phi[i])
            vaphi.append(va_br(phi[i], dim))
            vaphi2.append(vaphi[i]**2)
            vaphi4.append(vaphi[i]**4)
            gphi = grad_r(phi[i], dx, dim)
            phix.append(gphi[0])
            phiy.append(gphi[1])
        
            #add new row to matrices
            phix_ij.append([])
            phiy_ij.append([])
            psix_ij.append([])
            psiy_ij.append([])
            eta_ij.append([])
        
            #fill non-diagonal members of matrices
            for j in range(i):
                pxij = phix[i] - phix[j]
                pxij = 0.5*(pxij+np.roll(pxij, -1, 1))
                phix_ij[i].append(pxij)
                phix_ij[j].append(pxij)
                pyij = phiy[i] - phiy[j]
                pyij = 0.5*(pyij+np.roll(pyij, -1, 0))
                phiy_ij[i].append(pyij)
                phiy_ij[j].append(pyij)
                psixij = q2q2*pxij-qq2*pyij
                psiyij = qq2*pxij+q2q2*pyij
                psix_ij[i].append(psixij)
                psix_ij[j].append(psixij)
                psiy_ij[i].append(psiyij)
                psiy_ij[j].append(psiyij)
                #due to divide-by-zero possibility, set equal to zero if equals nan
                eij = np.nan_to_num(1-3*gamma[i][j]+4*gamma[i][j]*(psixij**4+psiyij**4)/((pxij**2+pyij**2)**2))
                eta_ij[i].append(eij)
                eta_ij[j].append(eij)
                
            #fill diagonal entry of matrices
            phix_ij[i].append(zeros)
            phiy_ij[i].append(zeros)
            psix_ij[i].append(zeros)
            psiy_ij[i].append(zeros)
            eta_ij[i].append(zeros)
        
        
        phi2sum = np.sum(phi2, axis=0)
        phi4sum = np.sum(phi4, axis=0)
        vaphi2sum = np.sum(vaphi2, axis=0)
        vaphi4sum = np.sum(vaphi4, axis=0)
        #magic! the sum of all pairs of phi_i^2phi_j^2 can be found using the following trick
        ssp2p2 = 0.5*(phi2sum**2-phi4sum) #cell
        issp2p2 = 1./ssp2p2 #convenience, since division is expensive and its used a lot
        issp2p2[np.isinf(issp2p2)] = 0
        vassp2p2 = 0.5*(vaphi2sum**2-vaphi4sum) #VERTEX
        ivassp2p2 = 1./vassp2p2 #convenience, since division is expensive and its used a lot
        ivassp2p2[np.isinf(ivassp2p2)] = 0
        #other sums are not so easy...
        #the format for these names is the abbreviation of the term
        #for example: s2e2npp2 is sum_{k!=j}_2epsilon^2*eta*phi_j*phi_k^2
        sswp2p2 = 0 #cell
        vasse2np2p2 = 0 #VERTEX
        vas2e2npp2 = [] #vector, for each phi_j, VERTEX
        s2wpp2 = [] #vector, for each phi_j, cell
        s2pp2 = [] #vector, for each phi_j, cell
        vas2pp2 = [] #vector, for each phi_j, VERTEX
        de2dpxj = [] #vector of de^2/dphi_{x,j}, for each phi_j, VERTEX
        de2dpyj = [] #vector of de^2/dphi_{y,j}, for each phi_j, VERTEX
        de2dq1 = 0 #vac
        de2dq4 = 0 #vac
        for i in range(len(phi)):
            vas2e2npp2.append(0)
            s2wpp2.append(0)
            de2dpxj.append(0)
            de2dpyj.append(0)
            s2pp2.append(2*phi[i]*(phi2sum-phi2[i]))
            vas2pp2.append(2*vaphi[i]*(vaphi2sum-vaphi2[i]))
            for j in range(i):
                sswp2p2 += _w[i][j]*phi2[i]*phi2[j]
                vasse2np2p2 += ebar2[i][j]*eta_ij[i][j]*vaphi2[i]*vaphi2[j]
                vas2e2npp2[i] += 2*ebar2[i][j]*eta_ij[i][j]*vaphi[i]*vaphi2[j]
                vas2e2npp2[j] += 2*ebar2[i][j]*eta_ij[i][j]*vaphi[j]*vaphi2[i]
                s2wpp2[i] += 2*_w[i][j]*phi[i]*phi2[j]
                s2wpp2[j] += 2*_w[i][j]*phi[j]*phi2[i]
                gp2 = (phix_ij[i][j]**2+phiy_ij[i][j]**2)
                gp4 = gp2**2
                gp8 = gp4**2
                psix3 = psix_ij[i][j]**3
                psiy3 = psiy_ij[i][j]**3
                gp2psi4 = gp2*(psix3*psix_ij[i][j]+psiy3*psiy_ij[i][j])
                dedpx = np.nan_to_num(16*gamma[i][j]*ebar2[i][j]*vaphi2[i]*vaphi2[j]*((gp4*(psix3*q2q2+psiy3*qq2)-gp2psi4*phix_ij[i][j])/gp8))
                dedpy = np.nan_to_num(16*gamma[i][j]*ebar2[i][j]*vaphi2[i]*vaphi2[j]*((gp4*(psiy3*q2q2-psix3*qq2)-gp2psi4*phiy_ij[i][j])/gp8))
                de2dpxj[i] += dedpx
                de2dpxj[j] += dedpx
                de2dpyj[i] += dedpy
                de2dpyj[j] += dedpy
                dpsixdq1 = 2*va_q1*phix_ij[i][j]-2*va_q4*phiy_ij[i][j]
                dpsiydq1 = 2*va_q1*phiy_ij[i][j]+2*va_q4*phix_ij[i][j]
                de2dq1 += np.nan_to_num(16*gamma[i][j]*ebar2[i][j]*vaphi2[i]*vaphi2[j]*(psix3*dpsixdq1+psiy3*dpsiydq1)/gp4)
                dpsixdq4 = -2*va_q4*phix_ij[i][j]-2*va_q1*phiy_ij[i][j]
                dpsiydq4 = -2*va_q4*phiy_ij[i][j]+2*va_q1*phix_ij[i][j]
                de2dq4 += np.nan_to_num(16*gamma[i][j]*ebar2[i][j]*vaphi2[i]*vaphi2[j]*(psix3*dpsixdq4+psiy3*dpsiydq4)/gp4)
    
        #convert nan values to zero, essentially eliminates this term if only 1 non-zero phase is present
        de2dq1 = np.nan_to_num(de2dq1*ivassp2p2)
        de2dq4 = np.nan_to_num(de2dq4*ivassp2p2)
        dwdpj = [] #one for each phi_j, cell
        de2dpj = [] #one for each phi_j, vac
        
        #compute e2 before vertex averaging, since we need its vertex value
        e2 = np.nan_to_num(vasse2np2p2*ivassp2p2)
    
        #vertex average arrays need to be averaged to find value on the cell
        de2dq1 = va_ul(de2dq1, dim)
        de2dq4 = va_ul(de2dq4, dim)
        
        
        for i in range(len(phi)):
            de2dpxj[i] = np.nan_to_num(de2dpxj[i]*ivassp2p2)
            de2dpyj[i] = np.nan_to_num(de2dpyj[i]*ivassp2p2)
            dwdpj.append(np.nan_to_num((s2wpp2[i]*ssp2p2-sswp2p2*s2pp2[i])*issp2p2*issp2p2))
            de2dpj.append(np.nan_to_num((vas2e2npp2[i]*vassp2p2-vasse2np2p2*vas2pp2[i])*ivassp2p2*ivassp2p2))
            de2dpj[i] = va_ul(de2dpj[i], dim)
        
        w = np.nan_to_num(sswp2p2*issp2p2)
        
        return e2, w, de2dpj, de2dpxj, de2dpyj, dwdpj, de2dq1, de2dq4
        

def renormalize(q1, q4):
    q = np.sqrt(q1*q1+q4*q4)
    return q1/q, q4/q

def loadArrays_nc(data_path, timestep):
    _q1 = np.load(root_folder+"/data/"+data_path+'/q1_'+str(timestep)+'.npy')
    _q4 = np.load(root_folder+"/data/"+data_path+'/q4_'+str(timestep)+'.npy')
    _c = []
    for i in range(len(components)-1):
        _c.append(np.load(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep)+'.npy'))
    _phi = np.load(root_folder+"/data/"+data_path+'/phi_'+str(timestep)+'.npy')
    return timestep, _phi, _c, _q1, _q4

def loadArrays_ncnp(data_path, timestep):
    data = np.load(root_folder+"/data/"+data_path+'/data_'+str(timestep)+'.npy')
    _q1 = data[0]
    _q4 = data[1]
    _phi = list(data[2:(2+len(phases))])
    _c = list(data[(2+len(phases)):(2+len(phases)+len(components))])
    return timestep, _phi, _c, _q1, _q4

def saveArrays_nc(data_path, timestep, phi, c, q1, q4):
    np.save(root_folder+"/data/"+data_path+'/phi_'+str(timestep), phi)
    for i in range(len(c)):
        np.save(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep), c[i])
    np.save(root_folder+"/data/"+data_path+'/q1_'+str(timestep), q1)
    np.save(root_folder+"/data/"+data_path+'/q4_'+str(timestep), q4)
    
def saveArrays_ncnp(data_path, timestep, phi, c, q1, q4):
    data = np.concatenate((np.expand_dims(q1, axis=0), np.expand_dims(q4, axis=0)), axis=0)
    data = np.concatenate((data, phi), axis=0)
    data = np.concatenate((data, c), axis=0)
    np.save(root_folder+"/data/"+data_path+'/data_'+str(timestep), data)

def applyBCs_nc(phi, c, q1, q4, nbc):
    if(nbc[0]):
        for i in range(len(c)):
            c[i][:,0] = c[i][:,1]
            c[i][:,-1] = c[i][:,-2]
        phi[:,0] = phi[:,1]
        phi[:,-1] = phi[:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
    if(nbc[1]):
        for i in range(len(c)):
            c[i][0,:] = c[i][1,:]
            c[i][-1,:] = c[i][-2,:]
        phi[0,:] = phi[1,:]
        phi[-1,:] = phi[-2,:]
        q1[0,:] = q1[1,:]
        q1[-1,:] = q1[-2,:]
        q4[0,:] = q4[1,:]
        q4[-1,:] = q4[-2,:]
        
def applyBCs_ncnp(phi, c, q1, q4, nbc):
    if(nbc[0]):
        for i in range(len(c)):
            c[i][:,0] = c[i][:,1]
            c[i][:,-1] = c[i][:,-2]
        for i in range(len(phi)):
            phi[i][:,0] = phi[i][:,1]
            phi[i][:,-1] = phi[i][:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
    if(nbc[1]):
        for i in range(len(c)):
            c[i][0,:] = c[i][1,:]
            c[i][-1,:] = c[i][-2,:]
        for i in range(len(phi)):
            phi[i][0,:] = phi[i][1,:]
            phi[i][-1,:] = phi[i][-2,:]
        q1[0,:] = q1[1,:]
        q1[-1,:] = q1[-2,:]
        q4[0,:] = q4[1,:]
        q4[-1,:] = q4[-2,:]

def coreSection(array, nbc):
    """
    Returns only the region of interest for plotting. 
    Removes the buffer cells used for Neumann Boundary Conditions
    """
    returnArray = array
    if(nbc[0]):
        returnArray = returnArray[:, 1:-1]
    if(nbc[1]):
        returnArray = returnArray[1:-1, :]
    return returnArray

def plotImages_nc(phi, c, q4, nbc, data_path, step):
    """
    Plots the phi (order), c (composition), and q4 (orientation component) fields for a given step
    Saves images to the defined path
    """
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('rgb', colors)
    colors2 = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
    cm2 = LinearSegmentedColormap.from_list('rgb', colors2)

    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = 4, 4
    plt.title('phi')
    cax = plt.imshow(coreSection(phi, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(phi), np.max(phi)])
    plt.savefig(root_folder+"/data/"+data_path+'/phi_'+str(step)+'.png')
    for i in range(len(c)):
        fig, ax = plt.subplots()
        plt.title('c_'+components[i])
        cax = plt.imshow(coreSection(c[i], nbc), cmap=cm)
        cbar = fig.colorbar(cax, ticks=[np.min(c[i]), np.max(c[i])])
        plt.savefig(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(step)+'.png')
    c_N = 1-np.sum(c, axis=0)
    fig, ax = plt.subplots()
    plt.title('c_'+components[len(c)])
    cax = plt.imshow(coreSection(c_N, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c_N), np.max(c_N)])
    plt.savefig(root_folder+"/data/"+data_path+'/c'+str(len(c)+1)+'_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    cax = plt.imshow(coreSection(q4, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(q4), np.max(q4)])
    plt.savefig(root_folder+"/data/"+data_path+'/q4_'+str(step)+'.png')
    
def plot1d(phi, c, q4, nbc, data_path, step):
    """
    Special case of plot2d, where one dimension is only 1 wide.
    Plots each field w.r.t. the length of the simulation, as a line plot
    """
    
    #first, orient the array so its shape is (1, length)
    if(coreSection(q4, nbc).shape[1] == 1):
        q4 = np.transpose(q4)
        phi = np.transpose(phi, (0, 2, 1))
        c = np.transpose(c, (0, 2, 1))
        
    xaxis = np.linspace(1, coreSection(q4, nbc).shape[1], num = coreSection(q4, nbc).shape[1])-1
    
    for i in range(len(phi)):
        fig, ax = plt.subplots()
        plt.title('phi_'+phases[i])
        plt.plot(xaxis, phi[i][0], dashes=[5, 3, i+1, 3], linewidth=i+1, color='red')
        plt.savefig(root_folder+"/data/"+data_path+'/phi'+str(i+1)+'_'+str(step)+'.png')
        
    for i in range(len(c)):
        fig, ax = plt.subplots()
        plt.title('c_'+components[i])
        plt.plot(xaxis, c[i][0], dashes=[5, 3, i+1, 3], linewidth=i+1, color='blue')
        plt.savefig(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(step)+'.png')
    
    c_N = 1-np.sum(c, axis=0)
    fig, ax = plt.subplots()
    plt.title('c_'+components[len(c)])
    plt.plot(xaxis, c_N[0], dashes=[5, 3, len(c)+1, 3], linewidth=len(c)+1, color='blue')
    plt.savefig(root_folder+"/data/"+data_path+'/c'+str(len(c)+1)+'_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    plt.plot(xaxis, q4[0], linewidth=1, color='black')
    plt.savefig(root_folder+"/data/"+data_path+'/q4_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('All Fields')
    legend = []
    for i in range(len(phi)):
        plt.plot(xaxis, phi[i][0], dashes=[5, 3, i+1, 3], linewidth=i+1, color='red')
        legend.append('phi_'+phases[i])
    for i in range(len(c)):
        plt.plot(xaxis, c[i][0], dashes=[5, 3, i+1, 3], linewidth=i+1, color='blue')
        legend.append('c_'+components[i])
    plt.plot(xaxis, c_N[0], dashes=[5, 3, len(c)+1, 3], linewidth=len(c)+1, color='blue')
    legend.append('c_'+components[len(c)])
    plt.plot(xaxis, np.absolute(q4[0]), linewidth=1, color='black')
    legend.append('q4')
    plt.legend(legend, bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(root_folder+"/data/"+data_path+'/all_'+str(step)+'.png', bbox_inches="tight")
    
    
def plot2d(phi, c, q4, nbc, data_path, step):
    """
    Plots the phi (order), c (composition), and q4 (orientation component) fields for a given step
    Saves images to the defined path
    """
    
    if(coreSection(q4, nbc).shape[0] == 1 or coreSection(q4, nbc).shape[1] == 1):
        plot1d(phi, c, q4, nbc, data_path, step)
        return
    
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('rgb', colors)
    colors2 = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
    cm2 = LinearSegmentedColormap.from_list('rgb', colors2)
    
    plt.rcParams['figure.figsize'] = 4, 4

    for i in range(len(phi)):
        fig, ax = plt.subplots()
        plt.title('phi_'+phases[i])
        cax = plt.imshow(coreSection(phi[i], nbc), cmap=cm2)
        cbar = fig.colorbar(cax, ticks=[np.min(phi[i]), np.max(phi[i])])
        plt.savefig(root_folder+"/data/"+data_path+'/phi'+str(i+1)+'_'+str(step)+'.png')
    for i in range(len(c)):
        fig, ax = plt.subplots()
        plt.title('c_'+components[i])
        cax = plt.imshow(coreSection(c[i], nbc), cmap=cm)
        cbar = fig.colorbar(cax, ticks=[np.min(c[i]), np.max(c[i])])
        plt.savefig(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(step)+'.png')
    c_N = 1-np.sum(c, axis=0)
    fig, ax = plt.subplots()
    plt.title('c_'+components[len(c)])
    cax = plt.imshow(coreSection(c_N, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c_N), np.max(c_N)])
    plt.savefig(root_folder+"/data/"+data_path+'/c'+str(len(c)+1)+'_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    cax = plt.imshow(coreSection(q4, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(q4), np.max(q4)])
    plt.savefig(root_folder+"/data/"+data_path+'/q4_'+str(step)+'.png')
    
def npvalue(var, string, tdb):
    """
    Returns a numpy float from the sympy expression gotten from pycalphad
    Reason: some numpy functions (i.e. sqrt) are incompatible with sympy floats!
    """
    return sp.lambdify(var, tdb.symbols[string], 'numpy')(1000)

def get_tdb_path_for_sim(data_path):
    with open(root_folder+"/data/"+data_path+"/info.txt") as search:
        lines = search.readlines()
        for line in lines:
            line = line.rstrip()
            if re.search('TDB File used', line):
                return re.split(': ', line)[1]
        #if it can't find this line, this is VERY BAD
        raise EOFError('Could not find TDB file used for simulation. This is very bad!')
        
def get_nbcs_for_sim(data_path):
    with open(root_folder+"/data/"+data_path+"/info.txt") as search:
        lines = search.readlines()
        for line in lines:
            line = line.rstrip()
            if re.search('Neumann Boundary Conditions', line):
                strings = re.split(': ', line)[1].strip("[]").split(',')
                nbcs = []
                for string in strings:
                    if string == "True":
                        nbcs.append(True)
                    else:
                        nbcs.append(False)
                return nbcs
        #if it can't find this line, this is VERY BAD
        raise EOFError('Could not find nbcs used for simulation. This is very bad!')
        
def multiObstacle(phi):
    #multiObstacle algorithm, derived from Cogswell2018
    #currently only for 2 and 3 phase models
    if(len(phi) > 3):
        raise ValueError('utils.multiObstacle is only coded for 2 and 3 phase models atm!')
    if(len(phi) == 3):
        bools = np.logical_and(phi[0]-phi[1] > 1,phi[0]-phi[2] > 1)
        phi[0] = np.where(bools, 1, phi[0])
        phi[1] = np.where(bools, 0, phi[1])
        phi[2] = np.where(bools, 0, phi[2])
        bools = np.logical_and(phi[1]-phi[0] > 1,phi[1]-phi[2] > 1)
        phi[0] = np.where(bools, 0, phi[0])
        phi[1] = np.where(bools, 1, phi[1])
        phi[2] = np.where(bools, 0, phi[2])
        bools = np.logical_and(phi[2]-phi[1] > 1,phi[2]-phi[0] > 1)
        phi[0] = np.where(bools, 0, phi[0])
        phi[1] = np.where(bools, 0, phi[1])
        phi[2] = np.where(bools, 1, phi[2])
        bools = phi[0] < 0
        phi[1] = np.where(bools, phi[1]+phi[0]/2, phi[1])
        phi[2] = np.where(bools, phi[2]+phi[0]/2, phi[2])
        phi[0] = np.where(bools, 0, phi[0])
        bools = phi[1] < 0
        phi[0] = np.where(bools, phi[0]+phi[1]/2, phi[0])
        phi[2] = np.where(bools, phi[2]+phi[1]/2, phi[2])
        phi[1] = np.where(bools, 0, phi[1])
        bools = phi[2] < 0
        phi[1] = np.where(bools, phi[1]+phi[2]/2, phi[1])
        phi[0] = np.where(bools, phi[0]+phi[2]/2, phi[0])
        phi[2] = np.where(bools, 0, phi[2])
    return phi
