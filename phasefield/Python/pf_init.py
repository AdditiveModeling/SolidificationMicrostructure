import numpy as np
import pf_utils as utils
import pf_engine as engine
import os
import sys

def preinitialize(sim_type, data_path, tdb_path):
    """
    Used in all initialize functions
    Creates a folder for the simulation that contains an "info.txt" file
    This file contains all relevant information about the simulation (parameters, simulation type, etc.)
    If an info.txt file already exists at the defined path, abort setup, and 
        notify user to choose a different location or delete the previous data
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    utils.load_tdb(tdb_path)
    if not engine.init_tdb_vars(utils.tdb):
        print("This TDB file doesn't have the required additional variables needed to run a phase field simulation!")
        # add required variable names! Important for debugging errors!
        print("Trying to run the simulation won't work!")
    if(os.path.isfile(utils.root_folder+"/data/"+data_path+"/info.txt")):
        print("A simulation has already been created at this path - aborting initialization!")
        print("Please choose a different path, or delete the previous simulation")
        return False
    if not os.path.isdir(utils.root_folder+"/data/"+data_path+"/"):
        os.makedirs(utils.root_folder+"/data/"+data_path+"/")
    info = open(utils.root_folder+"/data/"+data_path+"/info.txt", 'w')
    info.write("---Information about this Phase Field Simulation---\n\n")
    info.write("Simulation type: \n"+sim_type+"\n\n")
    info.write("Material Parameters: \n")
    info.write("    Anisotropy of S-L interfacial energy (y_e): "+str(engine.y_e)+"\n")
    info.write("    TDB File used: "+tdb_path+"\n")
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
    

def initializePlaneFront(data_path, tdb_path, lX, lY, c0=0):
    """
    Initializes a plane front simulation with a pre-generated fluctuation. 
    Used to compare whether the simulation parameters result in planar, cellular, or dendritic growth
    data_path: where the phi, c, q1, and q4 field data will be saved
    tdb_path: which TDB file will be used to run the simulation
    rX: Width of simulation region (x-axis)
    rY: Height of simulation region (y-axis)
    c0: initial composition array, if c0 = 0, autogenerate composition using equal fractions of each species
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Plane Front:\n    Size: ["+str(lX)+", "+str(lY)+"]\n    Neumann Boundary Conditions: [True, False]"
    if not preinitialize(sim_type, data_path, tdb_path):
        return False
    
    nbc = [True, False] #Neumann boundary conditions for x and y dimensions 
    #if false, its a periodic boundary instead

    shape = [lY, lX+2]
    dim = 2

    phi = np.zeros(shape)
    set_q = np.zeros(shape)
    set_q += 0*np.pi/16.
    q1 = np.cos(set_q)
    q4 = np.sin(set_q)
    c = []
    if c0 == 0:
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += 1./len(utils.components)
            c.append(c_i)
    elif(len(c0) == (len(utils.components)-1)):
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += c0[i]
            c.append(c_i)
    else:
        print("Mismatch between initial composition array length, and number of components!")
        print("c array must have "+str(len(utils.components)-1)+" values!")
        return False

    #initialize left side with a small solid region
    phi[:,0:5] = 1.
    
    #add small instability, will either disappear (stable planar growth) or grow (cellular/dendritic growth)
    for i in range((int)(lY/2-5), (int)(lY/2+5)):
        for j in range((int)(0), (int)(10)):
            if((i-lY/2)*(i-lY/2)+(j-5)*(j-5) < 25):
                phi[i][j] = 1.

    utils.applyBCs_nc(phi, c, q1, q4, nbc)
    utils.saveArrays_nc(data_path, 0, phi, c, q1, q4)
    return True
    
def initializeSeeds(data_path, tdb_path, lX, lY, nbcX, nbcY, numseeds, c0=0):
    """
    Initializes a simulation with several pre-generated seed crystals, of random orientation. 
    data_path: where the phi, c, q1, and q4 field data will be saved
    tdb_path: which TDB file will be used to run the simulation
    lX: Width of simulation region (x-axis)
    lY: Height of simulation region (y-axis)
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    numseeds: How many seed crystals to initialize
    c0: initial composition array, if c0 = 0, autogenerate composition using equal fractions of each species
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Multiple Seeds:\n    Size: ["+str(lX)+", "+str(lY)+"]\n    Neumann Boundary Conditions: ["+str(nbcX)+", "+str(nbcY)+"]\n    Number of Seeds: "+str(numseeds)
    if not preinitialize(sim_type, data_path, tdb_path):
        return False
    
    nbc = [nbcX, nbcY]
    shape = []
    dim = 2
    seeds = numseeds
    #this block initializes the size of the region, adding 2 if necessary to allow Neumann boundary conditions to work
    #because of how arrays are organized in python, this starts with Y (column #) then X (row #)
    if(nbcY):
        shape.append(lY+2)
    else:
        shape.append(lY)
    if(nbcX):
        shape.append(lX+2)
    else:
        shape.append(lX)

    phi = np.zeros(shape)
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(1*np.pi/8)
    q4 += np.sin(1*np.pi/8)
    c = []
    if c0 == 0:
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += 1./len(utils.components)
            c.append(c_i)
    elif(len(c0) == (len(utils.components)-1)):
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += c0[i]
            c.append(c_i)
    else:
        print("Mismatch between initial composition array length, and number of components!")
        print("c array must have "+str(len(utils.components)-1)+" values!")
        return False

    randAngle = np.random.rand(seeds)*np.pi/4
    randX = np.random.rand(seeds)*(lX-8)+4
    randY = np.random.rand(seeds)*(lY-8)+4
    for k in range(seeds):
        for i in range((int)(randY[k]-5), (int)(randY[k]+5)):
            for j in range((int)(randX[k]-5), (int)(randX[k]+5)):
                if((i-randY[k])*(i-randY[k])+(j-randX[k])*(j-randX[k]) < 25):
                    phi[i][j] = 1
                    q1[i][j] = np.cos(randAngle[k])
                    q4[i][j] = np.sin(randAngle[k])

    utils.applyBCs_nc(c, phi, q1, q4, nbc)
    utils.saveArrays_nc(data_path, 0, phi, c, q1, q4)
    return True
    
def initializeSeed(data_path, tdb_path, lX, lY, nbcX, nbcY, c0=0):
    """
    Initializes a simulation with a single seed crystal in the center, of random orientation. 
    data_path: where the phi, c, q1, and q4 field data will be saved
    tdb_path: which TDB file will be used to run the simulation
    lX: Width of simulation region (x-axis)
    lY: Height of simulation region (y-axis)
    nbcX: Whether Neumann boundary conditions are used along the x-axis. Otherwise, boundary is periodic 
    nbcY: Same as above, but for the y-axis. 
    c0: initial composition array, if c0 = 0, autogenerate composition using equal fractions of each species
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  Single Seed:\n    Size: ["+str(lX)+", "+str(lY)+"]\n    Neumann Boundary Conditions: ["+str(nbcX)+", "+str(nbcY)+"]"
    if not preinitialize(sim_type, data_path, tdb_path):
        return False
    
    nbc = [nbcX, nbcY]
    shape = []
    dim = 2
    seeds = 1
    if(nbcY):
        shape.append(lY+2)
    else:
        shape.append(lY)
    if(nbcX):
        shape.append(lX+2)
    else:
        shape.append(lX)

    phi = np.zeros(shape)
    q1 = np.zeros(shape)
    q4 = np.zeros(shape)
    q1 += np.cos(0*np.pi/8)
    q4 += np.sin(0*np.pi/8)
    c = []
    if c0 == 0:
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += 1./len(utils.components)
            c.append(c_i)
    elif(len(c0) == (len(utils.components)-1)):
        for i in range(len(utils.components)-1):
            c_i = np.zeros(shape)
            c_i += c0[i]
            c.append(c_i)
    else:
        print("Mismatch between initial composition array length, and number of components!")
        print("c array must have "+str(len(utils.components)-1)+" values!")
        return False
        
    randAngle = np.random.rand(seeds)*np.pi/4-np.pi/8
    randX = [lX/2]
    randY = [lY/2]
    for k in range(seeds):
        for i in range((int)(randY[k]-5), (int)(randY[k]+5)):
            for j in range((int)(randX[k]-5), (int)(randX[k]+5)):
                if((i-randY[k])*(i-randY[k])+(j-randX[k])*(j-randX[k]) < 25):
                    phi[i][j] = 1
                    q1[i][j] = np.cos(randAngle[k])
                    q4[i][j] = np.sin(randAngle[k])
                    
    utils.applyBCs_nc(phi, c, q1, q4, nbc)
    utils.saveArrays_nc(data_path, 0, phi, c, q1, q4)
    return True

def initialize1D(data_path, tdb_path, lX, interface, same_ori, num_components, c_a, c_b):
    """
    Initializes a simulation with a solid region on the left, and a liquid region on the right 
    data_path: where the phi, c_i, q1, and q4 field data will be saved
    tdb_path: which TDB file will be used to run the simulation
    lX: Width of simulation region
    interface: point at which second region begins
    same_ori: if true, both regions have same q values, otherwise offset by 45 degrees
    num_components: number of components to simulate
    c_a: array giving the for the first N-1 components values in the first region
    c_b: same, but in the second region
    
    Returns True if initialization is successful, False if not (folder already exists?)
    """
    
    sim_type = "  1-Dimension:\n    Size: ["+str(lX)+"]\n    c_a: "+str(c_a)+", c_b: "+str(c_b)
    if not preinitialize(sim_type, data_path, tdb_path):
        return False
    
    nbc = [True, False]
    shape = []
    dim = 2
    shape.append(1)
    shape.append(lX+2)

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
        utils.saveArrays(data_path, 0, phi, c, q1, q4)
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
        utils.saveArrays_3c(data_path, 0, phi, c1, c2, q1, q4)
    
    return True

if __name__ == '__main__':
    if len(sys.argv) == 1: #no additional arguments
        print("Phase Field Initialization Script")
        print("")
        print("To learn how to initialize a simulation, type \'python "+sys.argv[0]+" simulation_type\'")
        print("Available simulation_type values: plane, seed, seeds, 1d")
        
        
    elif sys.argv[1] == 'seed': #initialize single seed
        if len(sys.argv) == 2: #only does 'python pf_init.py seed'
            print("Format for initializing sample single seed simulation:")
            print("    python "+sys.argv[0]+" seed singleseedsimulation Ni-Cu_Ideal.tdb l:50,100 nbc:F,T c:0.40831")
            print(" ")
            print("Explanation of additional parameters:")
            print("    singleseedsimulation - name of the folder where data is stored")
            print("    Ni-Cu_Ideal.tdb - name of TDB file to use for the simulation")
            print("    l:50,100 - lengths of edges of simulation region, in # of cells. Optional argument, defaults to 200x200")
            print("    nbc:F,T - Neumann boundary conditions for x and y axes: (T)rue = NBC, (F)alse = periodic. Defaults to F,F")
            print("    c:0.40831 - initial composition of region. Must have # of values equal to N-1 components of the TDB file.")
            print("         Defaults to equal amounts of each component (in Ni-Cu, this would be 50% copper, 50% nickel)")
        elif len(sys.argv) < 4:
            print("Format requires a data_path and a tdb_path at minimum!")
        else:
            lengths = ["200", "200"]
            nbcs = [False, False]
            c0 = 0
            for i in range(len(sys.argv)-4):
                temp = sys.argv[i+4].split(':')
                if not (len(temp) == 2):
                    print("Bad format of optional argument! Given: "+sys.argv[i+4])
                    print("Must be of the form arg:value,value...,value")
                    print("arg types: l = cell lengths, nbc = boundary conditions, c = initial composition values")
                elif(temp[0] == 'l'):
                    lengths = temp[1].split(',')
                    if not len(lengths) == 2:
                        print("2D simulations only! Must have exactly 2 length arguments under l")
                        sys.exit()
                elif(temp[0] == 'nbc'):
                    nbcs = temp[1].split(',')
                    if not len(nbcs) == 2:
                        print("2D simulations only! Must have exactly 2 nbc arguments under nbc")
                        sys.exit()
                    if(nbcs[0] == 'F'):
                        nbcs[0] == False
                    else:
                        nbcs[0] == True
                    if(nbcs[1] == 'F'):
                        nbcs[1] == False
                    else:
                        nbcs[1] == True
                elif(temp[0] == 'c'):
                    values = temp[1].split(',')
                    c0 = []
                    for j in range(len(values)):
                        c0.append(float(values[j]))
            initializeSeed(sys.argv[2], sys.argv[3], int(lengths[0]), int(lengths[1]), nbcs[0], nbcs[1], c0)
            
        
    elif sys.argv[1] == 'seeds': #initialize multiple seeds
        if len(sys.argv) == 2: #only does 'python pf_init.py seeds'
            print("Format for initializing sample multiple seed simulation:")
            print("    python "+sys.argv[0]+" seeds multiseedsimulation Ni-Cu_Ideal.tdb l:50,100 nbc:F,T c:0.40831")
            print(" ")
            print("Explanation of additional parameters:")
            print("    multiseedsimulation - name of the folder where data is stored")
            print("    Ni-Cu_Ideal.tdb - name of TDB file to use for the simulation")
            print("    l:50,100 - lengths of edges of simulation region, in # of cells. Optional argument, defaults to 200x200")
            print("    nbc:F,T - Neumann boundary conditions for x and y axes: (T)rue = NBC, (F)alse = periodic. Defaults to F,F")
            print("    c:0.40831 - initial composition of region. Must have # of values equal to N-1 components of the TDB file.")
            print("         Defaults to equal amounts of each component (in Ni-Cu, this would be 50% copper, 50% nickel)")
            print("    s:20 - number of seeds to initialize in the simulation. Defaults to 10")
        elif len(sys.argv) < 4:
            print("Format requires a data_path and a tdb_path at minimum!")
        else:
            lengths = ["200", "200"]
            nbcs = [False, False]
            c0 = 0
            s = 10
            for i in range(len(sys.argv)-4):
                temp = sys.argv[i+4].split(':')
                if not (len(temp) == 2):
                    print("Bad format of optional argument! Given: "+sys.argv[i+4])
                    print("Must be of the form arg:value,value...,value")
                    print("arg types: l = cell lengths, nbc = boundary conditions, c = initial composition values, s = # of seeds")
                elif(temp[0] == 'l'):
                    lengths = temp[1].split(',')
                    if not len(lengths) == 2:
                        print("2D simulations only! Must have exactly 2 length arguments under l")
                        sys.exit()
                elif(temp[0] == 'nbc'):
                    nbcs = temp[1].split(',')
                    if not len(nbcs) == 2:
                        print("2D simulations only! Must have exactly 2 nbc arguments under nbc")
                        sys.exit()
                    if(nbcs[0] == 'F'):
                        nbcs[0] == False
                    else:
                        nbcs[0] == True
                    if(nbcs[1] == 'F'):
                        nbcs[1] == False
                    else:
                        nbcs[1] == True
                elif(temp[0] == 'c'):
                    values = temp[1].split(',')
                    c0 = []
                    for j in range(len(values)):
                        c0.append(float(values[j]))
                elif(temp[0] == 's'):
                    if not temp[1].isdigit():
                        print("Must be an integer number of seeds!")
                        sys.exit()
                    s = int(temp[1])
                    
            initializeSeeds(sys.argv[2], sys.argv[3], int(lengths[0]), int(lengths[1]), nbcs[0], nbcs[1], s, c0)
        
    elif sys.argv[1] == 'plane': #initialize plane-front growth
        if len(sys.argv) == 2: #only does 'python pf_init.py seed'
            print("Format for initializing sample single seed simulation:")
            print("    python "+sys.argv[0]+" plane planefrontsimulation Ni-Cu_Ideal.tdb l:50,100 c:0.40831")
            print(" ")
            print("Explanation of additional parameters:")
            print("    planefrontsimulation - name of the folder where data is stored")
            print("    Ni-Cu_Ideal.tdb - name of TDB file to use for the simulation")
            print("    l:50,100 - lengths of edges of simulation region, in # of cells. Optional argument, defaults to 200x200")
            print("    c:0.40831 - initial composition of region. Must have # of values equal to N-1 components of the TDB file.")
            print("         Defaults to equal amounts of each component (in Ni-Cu, this would be 50% copper, 50% nickel)")
            print("    Note: Neumann Boundary conditions are always T,F for plane front simulation!")
        elif len(sys.argv) < 4:
            print("Format requires a data_path and a tdb_path at minimum!")
        else:
            lengths = ["200", "200"]
            c0 = 0
            for i in range(len(sys.argv)-4):
                temp = sys.argv[i+4].split(':')
                if not (len(temp) == 2):
                    print("Bad format of optional argument! Given: "+sys.argv[i+4])
                    print("Must be of the form arg:value,value...,value")
                    print("arg types: l = cell lengths, nbc = boundary conditions, c = initial composition values")
                elif(temp[0] == 'l'):
                    lengths = temp[1].split(',')
                    if not len(lengths) == 2:
                        print("2D simulations only! Must have exactly 2 length arguments under l")
                        sys.exit()
                elif(temp[0] == 'c'):
                    values = temp[1].split(',')
                    c0 = []
                    for j in range(len(values)):
                        c0.append(float(values[j]))
            initializePlaneFront(sys.argv[2], sys.argv[3], int(lengths[0]), int(lengths[1]), c0)
        
    elif sys.argv[1] == '1d': #initialize 1-d test simulation
        #currently broken! need to fix later!
        if len(sys.argv) == 2: #only does 'python pf_init.py seed'
            print("Format for initializing sample single seed simulation:")
            print("    python "+sys.argv[0]+" seed singleseedsimulation Ni-Cu_Ideal.tdb l:50,100 nbc:F,T c:0.40831")
            print(" ")
            print("Explanation of additional parameters:")
            print("    singleseedsimulation - name of the folder where data is stored")
            print("    Ni-Cu_Ideal.tdb - name of TDB file to use for the simulation")
            print("    l:50,100 - lengths of edges of simulation region, in # of cells. Optional argument, defaults to 200x200")
            print("    nbc:F,T - Neumann boundary conditions for x and y axes: (T)rue = NBC, (F)alse = periodic. Defaults to F,F")
            print("    c:0.40831 - initial composition of region. Must have # of values equal to N-1 components of the TDB file.")
            print("         Defaults to equal amounts of each component (in Ni-Cu, this would be 50% copper, 50% nickel)")
        elif len(sys.argv) < 4:
            print("Format requires a data_path and a tdb_path at minimum!")
        else:
            lengths = ["200", "200"]
            nbcs = [False, False]
            c0 = 0
            for i in range(len(sys.argv)-4):
                temp = sys.argv[i+4].split(':')
                if not (len(temp) == 2):
                    print("Bad format of optional argument! Given: "+sys.argv[i+4])
                    print("Must be of the form arg:value,value...,value")
                    print("arg types: l = cell lengths, nbc = boundary conditions, c = initial composition values")
                elif(temp[0] == 'l'):
                    lengths = temp[1].split(',')
                    if not len(lengths) == 2:
                        print("2D simulations only! Must have exactly 2 length arguments under l")
                        sys.exit()
                elif(temp[0] == 'nbc'):
                    nbcs = temp[1].split(',')
                    if not len(nbcs) == 2:
                        print("2D simulations only! Must have exactly 2 nbc arguments under nbc")
                        sys.exit()
                    if(nbcs[0] == 'F'):
                        nbcs[0] == False
                    else:
                        nbcs[0] == True
                    if(nbcs[1] == 'F'):
                        nbcs[1] == False
                    else:
                        nbcs[1] == True
                elif(temp[0] == 'c'):
                    values = temp[1].split(',')
                    c0 = []
                    for j in range(len(values)):
                        c0.append(float(values[j]))
            initialize1D(sys.argv[2], sys.argv[3], int(lengths[0]), int(lengths[1]), nbcs[0], nbcs[1], c0)
        
    elif sys.argv[1] == '--help': #initialize plane-front growth
        print("Phase Field Initialization Script")
        print("")
        print("To learn how to initialize a simulation, type \'python "+sys.argv[0]+" [simulation_name]\'")
        print("Available simulation_name values: plane, seed, seeds, 1d")