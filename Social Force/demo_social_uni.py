# Authors:
#     Sylvain Faure <sylvain.faure@math.u-psud.fr>
#     Bertrand Maury <bertrand.maury@math.u-psud.fr>
#
#      cromosim/examples/micro/social/micro_social.py
#      python micro_social.py --json input.json
#
# License: GPL


import sys, os
import numpy as np
from cromosim import *
from cromosim.micro import *
from optparse import OptionParser
import json
from scipy.spatial import distance
import statistics as stat
from tqdm import tqdm
#plt.ion()
plt.ioff()
"""
    python3 micro_granular.py --json input.json
"""
parser = OptionParser(usage="usage: %prog [options] filename",version="%prog 1.0")
parser.add_option('--json',dest="jsonfilename",default="input_uni_social.json",type="string",
                  action="store",help="Input json filename")
opt, remainder = parser.parse_args()
#print("===> JSON filename = ",opt.jsonfilename)
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
        Ellipses used to build walls, [ [x_center,y_center, width, height, angle_in_degrees_anti-clockwise],... ]
    wall_polygons : list of numpy arrays
        Polygons used to build walls, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    wall_lines : list of numpy arrays
        Polylines used to build walls, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    door_lines: list of numpy arrays
        Polylines used to build doors, [ [[x0,x1,x2,...],[y0,y1,y2,...]],... ]
    seed: integer
        Random seed which can be used to reproduce a random selection if >0
    rmin: float
        Minimum radius for people
    rmax: float
        Maximum radius for people
    mass: float
        Mass of one person (typically 80 kg)
    tau: float
        (typically 0.5 s)
    F: float
        Coefficient for the repulsion force between individuals (typically 2000 N)
    kappa: float
        Stiffness constant to handle overlapping (typically 120000 kg s^-2)
    delta: float
        To maintain a certain distance from neighbors (typically 0.08 m)
    Fwall: float
        Coefficient for the repulsion force between individual and walls (typically 2000 N, like for F)
    lambda: float
        Directional dependence (between 0 and 1 = fully isotropic case)
    eta: float
        Friction coefficient (typically 240000 kg m^-1 s^-1)
    N: list
        Number of persons in each boxes
    init_people_box: list
        List of boxes to randomly position people at initialization, \
        [[xmin,xmax,ymin,ymax],...]
    exit_people_box:
        People outside this box will be deleted, [xmin,xmax,ymin,ymax]
    Tf: float
        Final time
    dt: float
        Time step
    drawper: integer
        The results will be displayed every "drawper" iterations
    dmax: float
        Maximum distance used to detect neighbors
    dmin: float
        Minimum distance allowed between individuals
    sensors: list of numpy array
        Segments through which incoming and outgoing flows are measured
        [ [x0,y0,x1,y1],... ]
    plot_people: boolean
        If true, people are drawn
    plot_contacts: boolean
        If true, active contacts between people are drawn
    plot_velocities: boolean
        If true, people velocities are drawn
    plot_paths: boolean
        If true, people paths are drawn
    plot_sensors: boolean
        If true, plot sensor lines on people graph and sensor data graph
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
N = sp.array(input["N"]).astype(int)
#print("N = ",N)
Np = N.sum()
rmin = input["rmin"]
rmax = input["rmax"]
mass = input["mass"]
tau = input["tau"]
F = input["F"]
kappa = input["kappa"]
delta = input["delta"]
Fwall = input["Fwall"]
lambda_ = input["lambda"]
eta = input["eta"]
init_people_box = input["init_people_box"]
exit_people_box = input["exit_people_box"]
Tf = input["Tf"]
dt = input["dt"]
drawper = input["drawper"]
dmax = input["dmax"]
dmin = input["dmin"]
sensors = input["sensors"]
plot_p = input["plot_people"]
plot_c = input["plot_contacts"]
plot_v = input["plot_velocities"]
plot_pa = input["plot_paths"]
plot_s = input["plot_sensors"]
linewidth = input["line_width"]
filename = input["filename"]
vmax = input["v_max"]
plot = input["plot"]
ferma_prof = input["ferma_prof"]
total = input["total_it"]
'''print("===> Number of persons = ",Np)
print("===> Final time, Tf = ",Tf)
print("===> Time step, dt = ",dt)
print("===> To draw the results each drawper iterations, drawper = ",drawper)
print("===> Maximal distance to find neighbors, dmax = ",dmax,", example : 2*dt")
print("===> Minimal distance between persons, dmin = ",dmin)'''

"""
    Build the Domain
"""

## To create an Domain object
dom = Domain(name=name, pixel_size=px, width=width, height=height, vmax = vmax)
## To add lines : Line2D(xdata, ydata, linewidth)
for xy in wall_lines:
    line = Line2D( xy[0],xy[1], linewidth=linewidth)
    dom.add_wall(line)
## To add doors :
for xy in door_lines:
    line = Line2D( xy[0],xy[1], linewidth=linewidth)
    dom.add_door(line)
## To build the domain : background + shapes
dom.build_domain()
## To compute the distance to the walls
dom.compute_wall_distance()
## To compute the desired velocity
dom.compute_desired_velocity()
## To show the domain dimensions
'''
print("===> Domain : ",dom)
print("===> Wall lines : ",wall_lines)
print("===> Door lines : ",door_lines)

plt.ioff()
dom.plot()
plt.show()
'''
"""
    Initialization
"""
'''
f = open(filename, "w")
m_h = height * px
m_w = width * px
a = m_w * m_h
density = Np / a
f.write("Input: " + str(filename) + "\n")
f.write("Velocità: " + str(vmax) + "\n")
f.write("density: " + str(density) + "\n")
f.write("Np: " + str(Np) + "\n")
f.write("dt: " + str(dt) + "\n\n")
'''
mean = 0
times = []
c_flows = 0
flow_mean_total = 0
it = 1

#door_center = (stat.mean(door_lines[0][0]), stat.mean(door_lines[0][1]))

doors_center = []
for door in door_lines:
    doors_center.append((stat.mean(door[0]), stat.mean(door[1])))

time_single_unit = {}
exit_single_unit = {}
custom_area = [[3.9, 3.9], [4.8, 4.5]]
count_zoneB = []
count_zoneA = []
velocities_iter = []
evacuation_time = []
velocities_mean = []
flow_iter = []
inc = 0
iter = 0
repeat = False
for i in range(1, Np+1):
    time_single_unit[i] = []
    exit_single_unit[i] = []
    for j in range(0, len(door_lines)):
        exit_single_unit[i].append(0)


flows = []
flow_cum = []
people_dist = {}
people_pos = {}
people_door_dist = {}
people_velocities = {}
flows_doors = {}#flow per ogni porta
count_exit_for_door = []
for i in range(0, len(door_lines)): 
        flows_doors[i] = []#per ogni porta(chiave del dict) ho  un vettore con tutti i suoi flow
        count_exit_for_door.append(0)#conta quanti ne escono per ogni porta ad ogni secondo
        #density_near_doors[i] = []#densità intorno alla porta per ogni porta(come chiave)
        #count_p[i] = 0
    ## Current time
if (seed<0):
    seed = sp.random.RandomState()
## Current time
t = 0.0
people, people_init_box_id, rng = people_initialization(N, init_people_box, dom,
                                                        dt, rmin, rmax, dmin=dmin,
                                                        seed=seed)
people_ids = people[:,3]
for p in people:
    people_door_dist[p[3]] = [[distance.euclidean((p[0],p[1]), door_center)] for door_center in doors_center]
    people_velocities[p[3]] = [0]
    people_pos[p[3]] = [(p[0], p[1])]
    people_dist[p[3]] = 0
## Array to store the results : all the people coordinates for all times
Np = people.shape[0]
Uold = sp.zeros((Np,2))
Np_init = Np
people_id = sp.arange(Np)
results = sp.copy(people[:,:2]).reshape((Np,2,1))

## Array to store sensor data : time dir pts[2] for each sensor line
if (len(sensors)>0):
    sensor_data = sp.zeros((Np,4,len(sensors)))

count_p_custom_area = []#conta persone nella zona delimitata da custom_area
count_p_cum = []
"""
    Main loop
"""
cc = 0
counter = 0
int_t = int(t)
N_old = Np
tot_vel = []
i = 0
time = {}
ts = {}
people_velocities = {}
velocities = []
#print("\nStep_" + str(iter + 1) + " di " + str(total))
for p in people[:,3]:
    ts[p] = [0]
    people_velocities[p] = [0]
 
if(plot):
    dir = prefix +'iter' + str(it)
    os.makedirs(dir)
while (t<Tf):#continua fino allo scadere del tempo
    contacts = compute_contacts(dom, people, dmax)
    I, J, Vd = compute_desired_velocity(dom, people)

    
    if ((cc>=drawper) or (counter==0)):
        #print("===> time = ",t," number of persons = ",Np)
        if(plot == True):
            
            plot_people(10, dom, people, contacts, Vd, people[:,3], time=t,
                            plot_people=plot_p, plot_contacts=plot_c,
                            plot_velocities=plot_v, plot_paths=plot_pa,paths=results,
                            plot_sensors=plot_s, sensors=sensors,
                            savefig=True, filename= dir + '/fig_'+ \
                            str(counter).zfill(6)+ '_' + str(dt) + '.png')
            #plot_grafici(20, people, ts, people_door_dist, t, people_velocities, people_pos)
            plt.pause(0.01)
            if (t>0):
                for i, s in enumerate(sensors):
                    plot_sensor_data(30+i, sensor_data[:,:,i], t, savefig=True,
                            filename=prefix+'sensor_'+str(i)+'_'+str(counter)+'.png')
                    plt.pause(0.01)
        cc = 0
    Forces = compute_forces(F, Fwall, people, contacts, Uold, Vd, lambda_, delta, kappa, eta)
    U = dt*(Vd-Uold)/tau + Uold + dt*Forces/mass

    '''if ferma_prof:
        if(Np > 1 and people[-1,0] >= doors_center[0][0]-1):
            U[-1, :] = 0
        else:
            U[-1, :] = dt*(Vd[-1,:]-Uold[-1,:])/tau + Uold[-1,:] + dt*Forces[-1,:]/mass[1]
    '''
    people = move_people(t, dt, people, U)
    t += dt
    for p in people:
        i = 0
        for door_center in doors_center:
            dist = distance.euclidean((p[0],p[1]), door_center)
            #people_velocities[p[3]].append(people_door_dist[p[3]][-1]-dist)/t)
            people_door_dist[p[3]][i].append(dist)
            i += 1
        ts[p[3]].append(t)
        people_dist[p[3]] += distance.euclidean(people_pos[p[3]][-1], (p[0], p[1]))
        people_pos[p[3]].append((p[0], p[1]))
    t -= dt
    if len(U > 0):
        for p in range(0, len(people_id)):
            #calcolo accellerazione di ogni persona
            vel = pow((U[p][0]*U[p][0]) + (U[p][1]*U[p][1]), 1/2)
            if p in people_velocities.keys():
                people_velocities[people_id[p]].append(vel)
            else:
                people_velocities[people_id[p]] = [vel] 

        for u in U:
            vel = pow((u[0]*u[0]) + (u[1]*u[1]), 1/2)
            velocities.append(vel)

        tot_vel.append(sum(velocities)/len(velocities))
    people, U, [people_id], id_exit = exit_door(2*dom.pixel_size, dom, people, U, doors_center, arrays=[ people_id])
    
    for id in id_exit:
        time_single_unit[id].append(t)
        exit_single_unit[id][id_exit[id]] += 1
        count_exit_for_door[id_exit[id]] += 1
    
    
    ## Store people positions in the result array (used to draw people paths)
    tmp = 1e99*sp.ones((Np_init,2))
    tmp[people_id,:] = people[:,:2]
    results = sp.concatenate((results,tmp.reshape((Np_init,2,1))), axis=2)
    for id in id_exit:
        time[id] = t
    t += dt
    Uold = U
    cc += 1
    counter += 1
    
    if int_t < int(t):#scatta nuovo secondo
        #aggiorno il flow rate secondo per secondo
        flow = N_old - Np
        N_old = Np
        int_t = int(t)
        flows.append(flow)
        #vel_to_plot.append(sum(velocities)/len(velocities))

        

        
        #salvo il flow rate per ogni porta e la densità vicino ad ogni porta
        for i in range(0, len(count_exit_for_door)):
            flows_doors[i].append(count_exit_for_door[i])
            #density_near_doors[i].append(count_p[i] / area_near_doors_size[i])

        
        #azzerro i contatore temporanei
        for i in range(0, len(door_lines)):
            count_exit_for_door[i] = 0
            #count_p[i] = 0

    Np = people.shape[0]
    if(t > 80):
        inc += 1
        #f.write("=========> scemo incastrato" )
        #print("=========> scemo incastrato" )
        plot_people(10, dom, people, contacts, Vd, people[:,3], time=t,
                            plot_people=plot_p, plot_contacts=plot_c,
                            plot_velocities=plot_v, plot_paths=plot_pa,paths=results,
                            plot_sensors=plot_s, sensors=sensors,
                            savefig=True, filename="incastrato" + str(inc) + ".png")
        repeat = True
        break
    if (Np == 0):
        flow = N_old - Np
        flows.append(flow)
        for i in range(0, len(count_exit_for_door)):
            flows_doors[i].append(count_exit_for_door[i])

        #if debug: print("===> time = ",int_t," flow = ", flow, "flow per porta = ", count_exit_for_door, "number of persons = ", Np)

        for i in range(0, len(door_lines)):
            count_exit_for_door[i] = 0
        #if debug: print("END... Nobody here !")

        break
'''
iter += 1
it += 1
flow_rate = sum(flows) / len(flows)
flow_mean_total += flow_rate
c_flows += 1
mean = mean + t
f.write("velocity" + str(c_flows) + " = " + str(sum(tot_vel)/len(tot_vel)) + "m/s\n")
print("velocity" + str(c_flows) + " = " + str(sum(tot_vel)/len(tot_vel)) + "m/s")
#velocities_iter.append(sum(tot_vel)/len(tot_vel))
velocities_iter.append(tot_vel)
print("evacuation time = ", t, "s")
print("path length = ", t * (sum(tot_vel)/len(tot_vel)), "m")
f.write("evacuation time_" + str(c_flows) +": " + str(t) + "\n")
evacuation_time.append(t)
velocities_mean.append(sum(tot_vel)/len(tot_vel))
f.write("flow_rate_" + str(c_flows) + ": " + str(flow_rate) + "\n")
f.write("flows" + str(flows) + "\n")
flow_iter.append(flows)
f.write("flows per ogni porta: \n")
f.write(str(flows_doors) + "\n")
#f.write("persona nell'area a scelta: \n")
#f.write(str(count_p_custom_area) + "\n")
f.write("Media lunghezza cammini: " + str(stat.mean([v for v in people_dist.values()])) + "\n\n")
#count_zoneB.append(count_p_custom_area)
#count_zoneA.append(flows)
times.append(t)

#plt.ioff()
#plt.show()

flow_mean_total = flow_mean_total / c_flows
mean = mean/ total
f.write("all times\n")
f.write(str(times))
f.write("\n")
f.write("mean ev: " + str(mean) + "\n")
f.write("std ev: " + str(np.std(times)) + "\n")
#f.write("mean door flow rate: " + str(flow_mean_total) + "\n")
#f.write("mean desk flow rate: " + str(stat.mean(count_p_custom_area)) + "\n")
f.write("\n\n" + "Velocità medie :" + "\n" + str(velocities_mean) + "\n")
f.write("Flussi ad ogni dt :" + "\n" + str(flow_iter))
#print(time_single_unit)
for id in time_single_unit:
    mean = sum(time_single_unit[id]) / len(time_single_unit[id])
    time_single_unit[id] = [mean]
    time_single_unit[id].extend(exit_single_unit[id])
    if len(door_lines) == 1:
        time_single_unit[id].append(0)

f.write("\ntime_single_unit\n")
f.write(str(time_single_unit))

'''
sys.exit()