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
    # will automatically update "phases" in multiphase model. For now, phases is hardcoded
    
    #update components
    components = list(tdb.elements)
    components.sort()
    return True

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

def loadArrays_nc(data_path, timestep):
    _q1 = np.load(root_folder+"/data/"+data_path+'/q1_'+str(timestep)+'.npy')
    _q4 = np.load(root_folder+"/data/"+data_path+'/q4_'+str(timestep)+'.npy')
    _c = []
    for i in range(len(components)-1):
        _c.append(np.load(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep)+'.npy'))
    _phi = np.load(root_folder+"/data/"+data_path+'/phi_'+str(timestep)+'.npy')
    return timestep, _phi, _c, _q1, _q4

def saveArrays_nc(data_path, timestep, phi, c, q1, q4):
    np.save(root_folder+"/data/"+data_path+'/phi_'+str(timestep), phi)
    for i in range(len(c)):
        np.save(root_folder+"/data/"+data_path+'/c'+str(i+1)+'_'+str(timestep), c[i])
    np.save(root_folder+"/data/"+data_path+'/q1_'+str(timestep), q1)
    np.save(root_folder+"/data/"+data_path+'/q4_'+str(timestep), q4)

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