# Authors:
#     Sylvain Faure <sylvain.faure@math.u-psud.fr>
#     Bertrand Maury <bertrand.maury@math.u-psud.fr>
#
#      cromosim/examples/cellular_automata/cellular_automata.py
#      python cellular_automata.py --json input.json
#
# License: GPL

import sys, os
import numpy as np
from cromosim import *
from cromosim.ca import *
from optparse import OptionParser
import json

"""
    python3 cellular_automata.py --json input.json
"""
parser = OptionParser(usage="usage: %prog [options] filename",version="%prog 1.0")
parser.add_option('--json',dest="jsonfilename",default="input_cat.json",type="string",
                  action="store",help="Input json filename")
parser.add_option('--vmax', dest = "v_max", default = 1, type = 'float', action="store", help="Vel max")
parser.add_option('--drawper', dest = "drawper", default = 20, type = 'int', action="store", help="Drawper")
#parser.add_option('--dt', dest = "dt", default = 0.3, type = 'float', action="store", help="Dt")
parser.add_option('--movd', dest = "movD", default = 'False', type = 'string', action="store", help="Movement")
parser.add_option('--Np', dest = "Np", default = 20, type = 'int', action="store", help="Np")
opt, remainder = parser.parse_args()
print("===> JSON filename = ",opt.jsonfilename)
with open(opt.jsonfilename) as json_file:
    input = json.load(json_file)


"""
    Get parameters from json file :

    name: string
        Domain name
    prefix: string
        Folder name to store the results
    background: string
        Image file used as background
    px: float
        Pixel size in meters (also called space step)
    width: integer
        Domain width (equal to the width of the background image)
    height: integer
        Domain height (equal to the height of the background image)
    wall_lines : list of numpy arrays
        Polylines used to build walls, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    wall_ellipses : list of numpy arrays
        Ellipses used to build walls, [ [x_center,y_center, width, height, \
        angle_in_degrees_anti-clockwise],... ]
    wall_polygons : list of numpy arrays
        Polygons used to build walls, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    wall_lines : list of numpy arrays
        Polylines used to build walls, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    door_lines: list of numpy arrays
        Polylines used to build doors, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    update_strategy: string
            Rules used to move the individuals : 'sequential' or 'parallel'
    seed: integer
        Random seed which can be used to reproduce a random selection if >0
    Np: integer
        Number of persons
    kappa: float
        Parameter for Static Floor Field
    Tf: float
        Final time
    dt: float
        Time step
    drawper: integer
        The results will be displayed every "drawper" iterations
"""

name = input["name"]
prefix = input["prefix"]
if not os.path.exists(prefix):
    os.makedirs(prefix)
background = input["background"]
px = input["px"]
width = input["width"]
height = input["height"]
wall_lines = input["wall_lines"]
wall_ellipses = input["wall_ellipses"]
wall_polygons = input["wall_polygons"]
door_lines = input["door_lines"]
seed = input["seed"]
update_strategy = input["update_strategy"]
kappa = input["kappa"]
Tf = input["Tf"]
dt = input["dt"]
drawper = input["drawper"]
filename = input["filename"]
persone = input["people_pos"][str(opt.Np)]
numeroIter = input["numeroIterazioni"]
plot = input["plot"]
Np = len(persone)
if len(persone) == 0:
    Np = input ['Np']
print(Np)
idProf = Np - 1
if idProf == "None":
    idProf= None
else:
	idProf = int(idProf)
print("profid",idProf)
movD = input["movimentoDiagonali"]
friction = input["friction"]
nExit = len(door_lines)
order = input['order'] 

#opzioni da terminale
dt = 0.3 / opt.v_max
movD = eval(opt.movD)
drawper = opt.drawper / 10


'''print("===> Number of persons = ",Np)
print("===> Final time, Tf = ",Tf)
print("===> Time step, dt = ",dt)
print("===> To draw the results each drawper iterations, drawper = ",drawper)'''

"""
    Build the Domain
"""

## To create an Domain object
dom = Domain(name=name, pixel_size=px, width=width, height=height)

## To add lines : Line2D(xdata, ydata, linewidth)
for xy in wall_lines:
    line = Line2D( xy[0],xy[1], linewidth=1)
    dom.add_wall(line)
## To add doors :
for xy in door_lines:
    line = Line2D( xy[0],xy[1], linewidth=1)
    dom.add_door(line)
## To build the domain : background + shapes
dom.build_domain()
## To compute the distance to the walls
dom.compute_wall_distance()

## To compute the desired velocity
dom.compute_desired_velocity()
## To show the domain dimensions
'''print("===> Domain : ",dom)
print("===> Wall lines : ",wall_lines)
print("===> Door lines : ",door_lines)'''


"""
    Initialization
"""
f = open('plot_data.txt', 'w')
f.write("Dt: " + str(dt)+ "\n")
'''f = open(filename, "w")
f.write("NP: " + str(Np)+ "\n")
f.write("Dt: " + str(dt)+ "\n")
f.write("strategy: " + str(update_strategy)+ "\n")
f.write("Movimento Diagonale: " + str(movD)+ "\n")
f.write("Prof: " + str(idProf)+ "\n" + "\n")
'''
mean = 0
times = []


c_flows = 0
flow_mean_total = 0

#dizionario chiave id persona, valore media tempi tutte iterazioni
meanTimeForPositions = {}
for key in range(1,Np+1):
    meanTimeForPositions[key] = [0,0,0]

for i in range(0, numeroIter):
    flows = []
    cc = 0
    iter = 0
    Npappo = Np
    ## Current time
    t = 0
    int_t = int(t)
    N_old = Npappo
    people = sp.ma.MaskedArray(sp.zeros((height,width),dtype=int), mask=dom.mask)

    ## Number of cells
    Nc = height*width - dom.mask_id[0].shape[0]
    print("===> Number of cells = ",Nc)

    ## People initialisation taking to account masked positions
    rng = sp.random.RandomState()
    
    if len(persone) == 0: #check per vedere se posizioniare persone a caso o meno
        print("inizializzazione casuale")
        order = None
        if update_strategy == 'sequential_fixed':
            print("Impossibile avere ordine se posizionati a caso!!")
            break
        if (Nc<height*width):
            imin = dom.mask_id[0].min()+1
            imax = dom.mask_id[0].max()-1
            jmin = dom.mask_id[1].min()+1
            jmax = dom.mask_id[1].max()-1
        else:
            imin = 0; imax = height
            jmin = 0; jmax = width
        people.data[22,10] = 1	#addProf
        s = 1
        while (s != Np):			#posizioni random
            people.data[rng.randint(imin,imax+1,Np-s),rng.randint(jmin,jmax+1,Np-s)] = 1
            s = sp.sum(people)
        ind = sp.where(people==1)
        people_ij = -sp.ones((Np,2),dtype=int)# i(0<=i<=height-1) j(0<=j<=width-1)
        people_ij[:,0] = ind[0]
        people_ij[:,1] = ind[1]
        idProf = 0
        for k in range(0,len(people_ij)):
            if people_ij[k][0] == 22 and people_ij[k][1] == 10:
                idProf = k			#take profId
    else:
        people_ij = -sp.ones((Np,2),dtype=int)# i(0<=i<=height-1) j(0<=j<=width-1)	
        count = 0
        for persona in persone:		#posizioni fisse
            people.data[persona[0],persona[1]] = 1
            
            people_ij[count][0] = persona[0]
            people_ij[count][1] = persona[1] 
            count = count + 1			
    
    
    #print("===> Init : people_ij = ",people_ij)
    ## Array to store the results
    results = sp.copy(people_ij).reshape((Np,2,1))

    ## Static Floor Field
    weight = sp.exp(-kappa*dom.door_distance)

    rng = sp.random.RandomState()
    if (seed>0):
        rng = sp.random.RandomState(seed)
    print("===> seed  = ",rng.get_state()[1][0])

    #finchè non tutti sono evacuati
	
    exitPeopleTime = []
    exitPeople = []
	
    if plot:# and i+1 == numeroIter:
        print("ULTIMO")
        plot_people_according_to_current_door_distance(1, people, dom,
		savefig=True, filename=prefix+'/fig_'+str(iter).zfill(5)+'0.png')
	
    while (sp.sum(people)>0):
        if (update_strategy == "parallel"):
            people, people_ij = parallel_update(people, people_ij, weight,
                                                friction = friction, idProf = idProf, movD = movD)
        elif (update_strategy == "sequential"):
            people, people_ij = sequential_update(people, people_ij, weight,
                                                  shuffle='random', randomstate = rng, idProf = idProf, movD = movD)
        elif (update_strategy == "sequential_frozen"):
            people, people_ij = sequential_update(people, people_ij, weight,
                                                  shuffle='random_frozen', randomstate = rng, idProf = idProf, movD = movD)
        elif (update_strategy == "sequential_fixed"):
            people, people_ij = sequential_update(people, people_ij, weight,
                                                  shuffle='fixed', randomstate = rng, idProf = idProf, movD = movD, fixedOrder = order)
        else:
            print("Bad value for update strategy... EXIT")
            sys.exit()
        if int_t < int(t):
            flow = N_old - Npappo
            #print("===> time = ",int_t," flow = ",flow, "number of persons = ", Npappo)
            N_old = Npappo
            int_t = int(t)
            flows.append(flow)
        people, people_ij, newExit = exit(dom, people, people_ij,exitPeople,nExit)#quanti ancora dentro
        results = sp.concatenate((results,people_ij.reshape((Np,2,1))), axis=2)
        t += dt
        cc += 1
        iter += 1
        #print("========> time = ",t," number of persons = ",sp.sum(people))
        if plot:# and i+1 == numeroIter:
            if (cc>=drawper):
                plot_people_according_to_current_door_distance(1, people, dom,
                savefig=True, filename=prefix+'/fig_'+str(iter).zfill(5)+'0.png')
                #plot_people_according_to_initial_door_distance(2, people, dom, results)
                cc = 0
			
        for p in newExit:
            exitPeopleTime.append([p[0],p[1],t])		
            meanTimeForPositions[p[0]+1][0]= meanTimeForPositions[p[0]+1][0] + t        #tempo medio
            meanTimeForPositions[p[0]+1][p[1]]= meanTimeForPositions[p[0]+1][p[1]] + 1  #conta numero di volte che esce in uscita 1 o 2			
            exitPeople.append(p[0])
        Npappo = sp.sum(people)
        if (Npappo == 0):
            flow = N_old - Npappo
            flows.append(flow)
            #print("===> time = ",int_t," flow = ",flow, "number of persons = ", Npappo)
            print("END... Nobody here !")
            break
	
    flow_rate = sum(flows) / len(flows)
    flow_mean_total += flow_rate
    c_flows += 1
    mean = mean + t
    times.append(t)
    #f.write(str(t) + "\n")
    f.write("flows per ogni porta:@{0:" + str(flows) + "}\n\n")	
    #print(exitPeopleTime)

flow_mean_total = flow_mean_total / c_flows
mean = mean/numeroIter
print(mean)
for key in meanTimeForPositions.keys():
     meanTimeForPositions[key][0] = meanTimeForPositions[key][0]/ numeroIter
print(meanTimeForPositions)
f.write(str(times)+ "\n")
f.write("mean ev: " + str(mean) + "\n")
f.write("std: " + str(np.std(times)) + "\n")
f.write("time_single_unit:@" + str(meanTimeForPositions))

if plot:
    plot_people_paths(2, dt, px, people, dom, results,
              savefig=True, filename=prefix+'/fig_paths.png')
    #plt.show()

'''
plt.bar(list(meanTimeForPositions.keys()), meanTimeForPositions.values(), color='g')
plt.xlabel("Posizione nell'aula")
plt.ylabel("Tempo evacuazione medio(s)")
plt.title("Tempi medi evacuazione in base alla locazione della persona")
plt.show()
'''