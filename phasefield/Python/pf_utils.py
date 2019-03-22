import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pycalphad as pyc

#initialize TDB file in pycalphad, from up a dir and in the TDB folder
rootFolder = os.path.abspath(os.path.dirname(__file__) + '/..')
tdb = pyc.Database(rootFolder + '/TDB/Ni-Cu.tdb')
phases = ['LIQUID', 'FCC_A1']
components = ['CU', 'NI']

def compute_tdb_energy(temps, c, phase):
    flattened_c = c.flatten()
    flattened_expanded_c = np.expand_dims(flattened_c, axis=1)
    flattened_expanded_binary_c = np.concatenate((flattened_expanded_c, 1-flattened_expanded_c), axis=1)
    #offset composition, for computing slope of GM w.r.t. comp
    feb_c_offset = flattened_expanded_binary_c+np.array([0.0000001, -0.0000001])
    flattened_t = temps.flatten()
    GM = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=flattened_expanded_binary_c, broadcast=False).GM.values.reshape(c.shape)
    GM_2 = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=feb_c_offset, broadcast=False).GM.values.reshape(c.shape)
    return GM, (GM_2-GM)*(10000000.)

def compute_tdb_energy_3c(temps, c1, c2, phase):
    #the order is aluminum, copper, nickel in the TDB calculate function (alphabetical!)
    flattened_c1 = c1.flatten()
    f_expanded_c1 = np.expand_dims(flattened_c1, axis=1)
    flattened_c2 = c2.flatten()
    f_expanded_c2 = np.expand_dims(flattened_c2, axis=1)
    flattened_expanded_trinary_c = np.concatenate((1-f_expanded_c1-f_expanded_c2, f_expanded_c2, f_expanded_c1), axis=1)
    #offset composition, for computing slope of GM w.r.t. comp
    fet_c1_offset = flattened_expanded_trinary_c+np.array([-0.0000001, 0., 0.0000001])
    fet_c2_offset = flattened_expanded_trinary_c+np.array([-0.0000001, 0.0000001, 0.])
    flattened_t = temps.flatten()
    GM = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=flattened_expanded_trinary_c, broadcast=False).GM.values.reshape(c1.shape)
    GM_2 = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=fet_c1_offset, broadcast=False).GM.values.reshape(c1.shape)
    GM_3 = pyc.calculate(tdb, components, phase, P=101325, T=flattened_t, points=fet_c2_offset, broadcast=False).GM.values.reshape(c1.shape)
    return GM, (GM_2-GM)*(10000000.), (GM_3-GM)*(10000000.)

def load_tdb(path):
    global tdb
    global phases
    global components
    tdb = pyc.Database(rootFolder + '/' + path)
    
    #update phases
    # will automatically updates "phases" for multiphase model. For now, phases is hardcoded
    
    #update components
    firstphase = tdb.phases[next(iter(tdb.phases.keys()))]
    ic = iter(firstphase.constituents[0])
    comp_array = []
    for i in range(len(firstphase.constituents[0])):
        comp = next(iter(next(ic).constituents))
        comp_array.append(comp)
    components = comp_array

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

def loadArrays_3c(path, timestep):
    _q1 = np.load(path+'q1_'+str(timestep)+'.npy')
    _q4 = np.load(path+'q4_'+str(timestep)+'.npy')
    _c1 = np.load(path+'c1_'+str(timestep)+'.npy')
    _c2 = np.load(path+'c2_'+str(timestep)+'.npy')
    _phi = np.load(path+'phi_'+str(timestep)+'.npy')
    return timestep, _phi, _c1, _c2, _q1, _q4

def saveArrays(path, timestep, phi, c, q1, q4):
    np.save(path+'phi_'+str(timestep), phi)
    np.save(path+'c_'+str(timestep), c)
    np.save(path+'q1_'+str(timestep), q1)
    np.save(path+'q4_'+str(timestep), q4)
    
def saveArrays_3c(path, timestep, phi, c1, c2, q1, q4):
    np.save(path+'phi_'+str(timestep), phi)
    np.save(path+'c1_'+str(timestep), c1)
    np.save(path+'c2_'+str(timestep), c2)
    np.save(path+'q1_'+str(timestep), q1)
    np.save(path+'q4_'+str(timestep), q4)
    
def applyBCs(phi, c, q1, q4, nbc):
    if(nbc[0]):
        c[:,0] = c[:,1]
        c[:,-1] = c[:,-2]
        phi[:,0] = phi[:,1]
        phi[:,-1] = phi[:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
    if(nbc[1]):
        c[0,:] = c[1,:]
        c[-1,:] = c[-2,:]
        phi[0,:] = phi[1,:]
        phi[-1,:] = phi[-2,:]
        q1[0,:] = q1[1,:]
        q1[-1,:] = q1[-2,:]
        q4[0,:] = q4[1,:]
        q4[-1,:] = q4[-2,:]
        
def applyBCs_3c(phi, c1, c2, q1, q4, nbc):
    if(nbc[0]):
        c1[:,0] = c1[:,1]
        c1[:,-1] = c1[:,-2]
        c2[:,0] = c2[:,1]
        c2[:,-1] = c2[:,-2]
        phi[:,0] = phi[:,1]
        phi[:,-1] = phi[:,-2]
        q1[:,0] = q1[:,1]
        q1[:,-1] = q1[:,-2]
        q4[:,0] = q4[:,1]
        q4[:,-1] = q4[:,-2]
    if(nbc[1]):
        c1[0,:] = c1[1,:]
        c1[-1,:] = c1[-2,:]
        c2[0,:] = c2[1,:]
        c2[-1,:] = c2[-2,:]
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

def plotImages(phi, c, q4, nbc, path, step):
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
    plt.savefig(path+'phi_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('c')
    cax = plt.imshow(coreSection(c, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c), np.max(c)])
    plt.savefig(path+'c_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    cax = plt.imshow(coreSection(q4, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(q4), np.max(q4)])
    plt.savefig(path+'q4_'+str(step)+'.png')
    
def plotImages_3c(phi, c1, c2, q4, nbc, path, step):
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
    plt.savefig(path+'phi_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('c1')
    cax = plt.imshow(coreSection(c1, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c1), np.max(c1)])
    plt.savefig(path+'c1_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('c2')
    cax = plt.imshow(coreSection(c2, nbc), cmap=cm)
    cbar = fig.colorbar(cax, ticks=[np.min(c2), np.max(c2)])
    plt.savefig(path+'c2_'+str(step)+'.png')
    fig, ax = plt.subplots()
    plt.title('q4')
    cax = plt.imshow(coreSection(q4, nbc), cmap=cm2)
    cbar = fig.colorbar(cax, ticks=[np.min(q4), np.max(q4)])
    plt.savefig(path+'q4_'+str(step)+'.png')