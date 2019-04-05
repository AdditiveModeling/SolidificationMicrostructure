import numpy as np
import pf_utils as utils
import pf_engine as engine
import os

def preinitialize(sim_type, path, pathToTDB):
    """
    Used in all initialize functions
    Creates a folder for the simulation that contains an "info.txt" file
    This file contains all relevant information about the simulation (parameters, simulation type, etc.)
    If an info.txt file already exists at the defined path, abort setup, and 
        notify user to choose a different location or delete the previous data
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    utils.load_tdb(pathToTDB)
    if not engine.init_tdb_vars(utils.tdb):
        print("This TDB file doesn't have the required additional variables needed to run a phase field simulation!")
        # add required variable names! Important for debugging errors!
        print("Trying to run the simulation won't work!")
    if(os.path.isfile(path+"info.txt")):
        print("A simulation has already been created at this path - aborting initialization!")
        print("Please choose a different path, or delete the previous simulation")
        return False
    if not os.path.isdir(path):
        os.makedirs(path)
    info = open(path+"info.txt", 'w')
    info.write("---Information about this Phase Field Simulation---\n\n")
    info.write("Simulation type: \n"+sim_type+"\n\n")
    info.write("Material Parameters: \n")
    info.write("    Anisotropy of S-L interfacial energy (y_e): "+str(engine.y_e)+"\n")
    info.write("    TDB File used: "+pathToTDB+"\n")
    info.write("  Other: \n")
    info.write("    Interfacial thickness (d): "+str(engine.d)+"\n\n")
    info.write("Discretization Parameters: \n")
    info.write("    Number of dimensions (dim): "+str(engine.dim)+"\n")
    info.write("    Cell size (dx): "+str(engine.dx)+"\n")
    info.write("    Time step (dt): "+str(engine.dt)+"\n\n")
    info.write("Notes: \n")
    info.write("    Components used: "+str(utils.components)+"\n")
    info.write("    Units are cm, s, J, K, mol\n")
    info.write("    0 = liquid, 1 = solid, opposite of Warren1995\n\n")
    info.write("Logs of simulation runs: \n\n")
    info.close()
    return True
    

def initializePlaneFront(rX, rY, path, pathToTDB):
    """
    Initializes a plane front simulation with a pre-generated fluctuation. 
    Used to compare whether the simulation parameters result in planar, cellular, or dendritic growth
    rX: Width of simulation region
    rY: Height of simulation region
    path: where the phi, c, q1, and q4 field data will be saved
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Plane Front:\n    Size: ["+str(rX)+", "+str(rY)+"]"
    if not preinitialize(sim_type, path, pathToTDB):
        return False
    
    nbc = [True, False] #Neumann boundary conditions for x and y dimensions 
    #if false, its a periodic boundary instead

    shape = [rY, rX+2]
    dim = 2
    resX = rX+2 #Neumann boundary conditions on X axis - needs an extra cell on either side as a buffer to stor
    resY = rY

    c = np.zeros(shape)
    phi = np.zeros(shape)
    set_q = np.zeros(shape)
    set_q += 0*np.pi/16.
    q1 = np.cos(set_q)
    q4 = np.sin(set_q)
    c += 0.40831

    #initialize left side with a small solid region
    phi[:,0:5] = 1.
    
    #add small instability, will either disappear (stable planar growth) or grow (cellular/dendritic growth)
    for i in range((int)(resY/2-5), (int)(resY/2+5)):
        for j in range((int)(0), (int)(10)):
            if((i-resY/2)*(i-resY/2)+(j-5)*(j-5) < 25):
                phi[i][j] = 1.

    utils.applyBCs(phi, c, q1, q4, nbc)
    utils.saveArrays(path, 0, phi, c, q1, q4)
    return True
    
def initializeSeeds(rX, rY, nbcX, nbcY, numseeds, path, pathToTDB):
    """
    Initializes a simulation with several pre-generated seed crystals, of random orientation. 
    rX: Width of simulation region
    rY: Height of simulation region
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    numseeds: How many seed crystals to initialize
    path: where the phi, c, q1, and q4 field data will be saved
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Multiple Seeds:\n    Size: ["+str(rX)+", "+str(rY)+"]\n    Neumann Boundary Conditions: ["+str(nbcX)+", "+str(nbcY)+"]\n    Number of Seeds: "+str(numseeds)
    if not preinitialize(sim_type, path, pathToTDB):
        return False
    
    nbc = [nbcX, nbcY]
    shape = []
    dim = 2
    seeds = numseeds
    resX = rX
    resY = rY
    #this block initializes the size of the region, adding 2 if necessary to allow Neumann boundary conditions to work
    #because of how arrays are organized in python, this starts with Y (column #) then X (row #)
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

    utils.applyBCs(c, phi, q1, q4, nbc)
    utils.saveArrays(path, 0, phi, c, q1, q4)
    return True
    
def initializeSeed(rX, rY, nbcX, nbcY, path, pathToTDB):
    """
    Initializes a simulation with a single seed crystal in the center, of random orientation. 
    rX: Width of simulation region
    rY: Height of simulation region
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    path: where the phi, c, q1, and q4 field data will be saved
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Single Seed:\n    Size: ["+str(rX)+", "+str(rY)+"]\n    Neumann Boundary Conditions: ["+str(nbcX)+", "+str(nbcY)+"]"
    if not preinitialize(sim_type, path, pathToTDB):
        return False
    
    nbc = [nbcX, nbcY]
    shape = []
    dim = 2
    seeds = 1
    resX = rX
    resY = rY
    if(nbcY):
        shape.append(resY+2)
    else:
        shape.append(resY)
    if(nbcX):
        shape.append(resX+2)
    else:
        shape.append(resX)

    phi = np.zeros(shape)
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(0*np.pi/8)
    q4 += np.sin(0*np.pi/8)
    c = []
    if(len(utils.components) == 2):
        c1 = np.zeros(shape)
        c1 += 0.40831
        c.append(c1)
    #manually doing 3+ component for now, will have to rewrite for N component model
    elif(len(utils.components) == 3):
        c1 = np.zeros(shape)
        c1 += 0.01
        c2 = np.zeros(shape)
        c2 += 0.40831
        c.append(c1)
        c.append(c2)
        
    elif(len(utils.components) == 5):
        c1 = np.zeros(shape)
        c1 += 0.01
        c2 = np.zeros(shape)
        c2 += 0.01
        c3 = np.zeros(shape)
        c3 += 0.40831
        c4 = np.zeros(shape)
        c4 += 0.56169
        c.append(c1)
        c.append(c2)
        c.append(c3)
        c.append(c4)

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
                    
    utils.applyBCs_nc(phi, c, q1, q4, nbc)
    utils.saveArrays_nc(path, 0, phi, c, q1, q4)
    return True

def initialize1D(rX, interface, same_ori, num_components, c_a, c_b, path, pathToTDB):
    """
    Initializes a simulation with a solid region on the left, and a liquid region on the right 
    rX: Width of simulation region
    interface: point at which second region begins
    same_ori: if true, both regions have same q values, otherwise offset by 45 degrees
    num_components: number of components to simulate
    c_a: array giving the for the first N-1 components values in the first region
    c_b: same, but in the second region
    path: where the phi, c_i, q1, and q4 field data will be saved
    pathToTDB: where the TDB file for the thermodynamics can be found
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  1-Dimension:\n    Size: ["+str(rX)+"]\n    c_a: "+str(c_a)+", c_b: "+str(c_b)+"\n    Thermodynamics: "+pathToTDB
    if not preinitialize(sim_type, path, pathToTDB):
        return False
    
    nbc = [True, False]
    shape = []
    dim = 2
    resX = rX
    shape.append(1)
    shape.append(resX+2)

    phi = np.zeros(shape)
    phi += 1.
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(0*np.pi/8)
    q4 += np.sin(0*np.pi/8)
    if(len(utils.components) == 2):
        c = np.zeros(shape)
        c += c_a
        c[0, interface:] = c_b
        phi[0, interface:] = 0
        if not same_ori:
            q1[0, interface:] = np.cos(1*np.pi/8)
            q4[0, interface:] = np.sin(0*np.pi/8)
        utils.applyBCs(c, phi, q1, q4, nbc)
        utils.saveArrays(path, 0, phi, c, q1, q4)
    #manually doing 3-component for now, will have to rewrite for N component model
    elif(len(utils.components) == 3):
        c1 = np.zeros(shape)
        c1 += c_a[0]
        c2 = np.zeros(shape)
        c2 += c_a[1]
        c1[0, interface:] = c_b[0]
        c2[0, interface:] = c_b[1]
        phi[0, interface:] = 0
        if not same_ori:
            q1[0, interface:] = np.cos(1*np.pi/8)
            q4[0, interface:] = np.sin(0*np.pi/8)
        utils.applyBCs_3c(phi, c1, c2, q1, q4, nbc)
        utils.saveArrays_3c(path, 0, phi, c1, c2, q1, q4)
    
    return True