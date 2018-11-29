import numpy as np
from python_engine import *

def initializePlaneFront(rX, rY, path):
    """
    Initializes a plane front simulation with a pre-generated fluctuation. 
    Used to compare whether the simulation parameters result in planar, cellular, or dendritic growth
    rX: Width of simulation region
    rY: Height of simulation region
    path: where the phi, c, q1, and q4 field data will be saved
    """

    nbc = [False, True] #Neumann boundary conditions for y and x dimensions (due to the way arrays are organized). 
    #if false, its a periodic boundary instead

    shape = []
    dim = 2
    resY = rX
    resX = rY
    if(nbc[0]):
        shape.append(resY+2)
    else:
        shape.append(resY)
    if(nbc[1]):
        shape.append(resX+2)
    else:
        shape.append(resX)

    c = np.zeros(shape)
    phi = np.zeros(shape)
    set_q = np.zeros(shape)
    set_q += 0*np.pi/16.
    q1 = np.cos(set_q)
    q4 = np.sin(set_q)
    c += 0.40831

    phi[:,0:5] = 1.
    c[:,0:5] = 0.40831*0.3994/0.4668
    for i in range((int)(resY/2-5), (int)(resY/2+5)):
        for j in range((int)(0), (int)(10)):
            if((i-resY/2)*(i-resY/2)+(j-5)*(j-5) < 25):
                phi[i][j] = 1.
                c[i][j] = 0.40831*0.3994/0.4668

    applyBCs(phi, c, q1, q4, nbc)
    if not os.path.isdir(path):
        os.makedirs(path)
    saveArrays(path, 0, phi, c, q1, q4)
    
def initializeSeeds(rX, rY, nbcX, nbcY, numseeds, path):
    """
    Initializes a simulation with several pre-generated seed crystals, of random orientation. 
    rX: Width of simulation region
    rY: Height of simulation region
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    numseeds: How many seed crystals to initialize
    path: where the phi, c, q1, and q4 field data will be saved
    """
    nbc = [nbcY, nbcX]
    shape = []
    dim = 2
    seeds = numseeds
    resY = rX
    resX = rY
    if(nbcY):
        shape.append(resY+2)
    else:
        shape.append(resY)
    if(nbcX):
        shape.append(resX+2)
    else:
        shape.append(resX)

    c = np.zeros(shape)
    phi = np.zeros(shape)
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(1*np.pi/8)
    q4 += np.sin(1*np.pi/8)
    c += 0.40831

    randAngle = np.random.rand(seeds)*np.pi/4
    randX = np.random.rand(seeds)*(resX-8)+4
    randY = np.random.rand(seeds)*(resY-8)+4
    for k in range(seeds):
        for i in range((int)(randY[k]-5), (int)(randY[k]+5)):
            for j in range((int)(randX[k]-5), (int)(randX[k]+5)):
                if((i-randY[k])*(i-randY[k])+(j-randX[k])*(j-randX[k]) < 25):
                    phi[i][j] = 1
                    q1[i][j] = np.cos(randAngle[k])
                    q4[i][j] = np.sin(randAngle[k])

    applyBCs(c, phi, q1, q4, nbc)
    if not os.path.isdir(path):
        os.makedirs(path)
    saveArrays(path, 0, phi, c, q1, q4)
    
def initializeSeed(rX, rY, nbcX, nbcY, path):
    """
    Initializes a simulation with a single seed crystal in the center, of random orientation. 
    rX: Width of simulation region
    rY: Height of simulation region
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    path: where the phi, c, q1, and q4 field data will be saved
    """
    nbc = [nbcY, nbcX]
    shape = []
    dim = 2
    seeds = 1
    resY = rX
    resX = rY
    if(nbcY):
        shape.append(resY+2)
    else:
        shape.append(resY)
    if(nbcX):
        shape.append(resX+2)
    else:
        shape.append(resX)

    c = np.zeros(shape)
    phi = np.zeros(shape)
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(0*np.pi/8)
    q4 += np.sin(0*np.pi/8)
    c += 0.40831

    randAngle = np.random.rand(seeds)*np.pi/4-np.pi/8
    randX = [resX/2]
    randY = [resY/2]
    for k in range(seeds):
        for i in range((int)(randY[k]-5), (int)(randY[k]+5)):
            for j in range((int)(randX[k]-5), (int)(randX[k]+5)):
                if((i-randY[k])*(i-randY[k])+(j-randX[k])*(j-randX[k]) < 25):
                    phi[i][j] = 1
                    q1[i][j] = np.cos(randAngle[k])
                    q4[i][j] = np.sin(randAngle[k])

    applyBCs(c, phi, q1, q4, nbc)
    if not os.path.isdir(path):
        os.makedirs(path)
    saveArrays(path, 0, phi, c, q1, q4)