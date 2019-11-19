# Authors:
#     Sylvain Faure <sylvain.faure@math.u-psud.fr>
#     Bertrand Maury <bertrand.maury@math.u-psud.fr>
#
#     cromosim/examples/micro/granular/micro_granular.py
#     python micro_granular.py --json input.json
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
import math  
#plt.ion()

def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist 

parser = OptionParser(usage="usage: %prog [options] filename",version="%prog 1.0")
parser.add_option('--json',dest="jsonfilename",default="input_cat.json",type="string",
                  action="store",help="Input json filename")
parser.add_option('--vmax', dest = "v_max", default = 1, type = 'float', action="store", help="Vel max")
parser.add_option('--drawper', dest = "drawper", default = 20, type = 'int', action="store", help="Drawper")
parser.add_option('--Np', dest = "Np", default = 20, type = 'int', action="store", help="Np")
parser.add_option('--p_index', dest = "p_index", default = '-1', type = 'string', action="store", help="p_index")
parser.add_option('--dmin', dest = "dmin", default = '0', type = 'float', action="store", help="dmin")
opt, remainder = parser.parse_args()
print("JSON filename = ",opt.jsonfilename)
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
    seed: integer
        Random seed which can be used to reproduce a random selection if >0
    rmin: float
        Minimum radius for people
    rmax: float
        Maximum radius for people
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
if opt.p_index == '-1':
    p_indexs = [-1]
else: 
    p_indexs = [opt.p_index]
print(p_indexs)

for p_index in p_indexs:

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
    #N = sp.array(input["N"]).astype(int)
    #Np = N.sum()
    rmin = input["rmin"]
    rmax = input["rmax"]
    exit_people_box = input["exit_people_box"]
    Tf = input["Tf"]
    dt = input["dt"]
    #drawper = input["drawper"]
    dmax = input["dmax"]
    dmin = input["dmin"]
    sensors = input["sensors"]
    plot_p = input["plot_people"]
    plot_c = input["plot_contacts"]
    plot_v = input["plot_velocities"]
    plot_pa = input["plot_paths"]
    plot_s = input["plot_sensors"]
    plot = input["plot"]
    linewidth = input["line_width"]
    total = input["total_it"]
    filename = input["filename"]
    #vmax = input["v_max"]
    plot = input["plot"]
    ferma_prof = input["ferma_prof"]
    debug = input["debug"]

    #opzioni da terminale
    vmax = opt.v_max
    drawper = opt.drawper
    dmin = opt.dmin
    
    if opt.jsonfilename != 'granular/input_uni.json':
        Np = opt.Np
        N = [1 for i in range(0, Np)]
        N = sp.array(N).astype(int)
        init_people_box = sp.array(input["init_people_box"][str(Np)])
    else:
        N = sp.array(input["N"]).astype(int)
        Np = N.sum()
        init_people_box = sp.array(input["init_people_box"])
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
    '''print("===> Domain : ",dom)
    print("===> Wall lines : ",wall_lines)
    print("===> Door lines : ",door_lines)'''
    '''
    plt.ioff()
    dom.plot(id=1)
    #dom.plot_desired_velocity(id=2)
    plt.show()
    '''
    """
        Initialization
    """
    #scelgo solo alcune posizioni
    if p_index != -1:
        aus = p_index.split(" ")
        p_index = []
        for i in aus:
            if i != '':
                p_index.append(int(i))
        N_aus = []
        init_aus = []
        for i in p_index:
            N_aus.append(N[int(i) - 1])
            init_aus.append(init_people_box[int(i) - 1])

        N = sp.array(N_aus).astype(int)
        init_people_box = sp.array(init_aus)
        Np = N.sum()

    m_h = height * px
    m_w = width * px
    a = m_w * m_h
    density = Np / a
    f = open('plot_data.txt', 'w')
    #f = open(filename + "_" + str(vmax) + "_" + str(dt) + "_" + str(density) + "_" + str(rmin) + "_" + str(rmax) + "_" + str(p_index) + ".txt", "w")
    '''f.write("density: " + str(density) + "\n")
    f.write("Np: " + str(Np) + "\n")
    f.write("dt: " + str(dt) + "\n")
    f.write("dmax: " + str(dmax) + "\n")
    f.write("vmax: " + str(vmax) + "\n")
    f.write("r: (" + str(rmin) + "," + str(rmax) + ")\n")
    f.write("p_index: " + str(p_index) + "\n\n")'''
    print("vmax: " + str(vmax))
    print("density: " + str(density))
    print("r: (" + str(rmin) + "," + str(rmax) + ")")
    print("p_index: " + str(p_index))
    print("Np: " + str(Np))
    mean = 0
    times = []
    c_flows = 0
    flow_mean_total = 0
    it = 1
    #door_center = (stat.mean(door_lines[0][0]), stat.mean(door_lines[0][1]))
    doors_center = []
    for door in door_lines:
        doors_center.append((stat.mean(door[0]), stat.mean(door[1])))

    people_door_dist = {}
    people_velocities = {}
    people_pos = {}

    time_single_unit = {}
    exit_single_unit = {}

    #calcolo densità vicino alla porta
    area_near_doors = {}
    area_dim = 1
    area_off = 0.3
    area_near_doors_size = {}#dimensione in metri quadri dell'area vicino ad ogni porta

    for i in range(0, len(door_lines)):
        minx = min(door_lines[i][0])
        maxx = max(door_lines[i][0])
        miny = min(door_lines[i][1])
        maxy = max(door_lines[i][1])
        #l'area è [[xp1, yp1], [xp2, yp2]]
        #p1: in basso a destra dell'area
        #p2: in alto a destra dell'area
        if minx == maxx:#in verticale
            with_mid = (width * px) / 2
            if minx < with_mid:#parete sinistra
                if debug: print("verticale a sinistra")
                area = [[minx, miny - area_off], [maxx + area_dim, maxy + area_off]]
            else:
                if debug: print("verticale a destra")
                area = [[minx - area_dim, miny - area_off], [maxx, maxy + area_off]]
        if miny == maxy:#in verticale
            height_mid = (height * px) / 2
            if miny < height_mid:#parete sinistra
                if debug: print("orizzontale in basso")
                area = [[minx - area_off, miny], [maxx + area_off, maxy + area_dim]]
            else:
                if debug: print("orizzontale in alto")
                area = [[minx - area_off, miny - area_dim], [maxx + area_off, maxy]]
        w_area = area[1][0] - area[0][0]
        h_area = area[1][1] - area[0][1]
        area_near_doors_size[i] = w_area * h_area
        area_near_doors[i] = area

    if debug: 
        print(area_near_doors)
        print(area_near_doors_size)

    for i in range(1, Np + 1):
        time_single_unit[i] = []
        exit_single_unit[i] = []
        for j in range(0, len(door_lines)):
            exit_single_unit[i].append(0)

    #persone ad ogni istante in un area a scelta dell'aula
    custom_area = [[3.9, 3.9], [4.8, 4.5]]

    for i in range(0, total):
        instants = []
        flows = []
        flows_doors = {}#flow per ogni porta
        count_exit_for_door = []
        density_near_doors = {}
        count_p = {}#ad ogni secondo conto quanti ce ne sono nell'area di ogni porta
        count_p_custom_area = []#conta persone nella zona delimitata da custom_area
        for i in range(0, len(door_lines)): 
            flows_doors[i] = []#per ogni porta(chiave del dict) ho  un vettore con tutti i suoi flow
            count_exit_for_door.append(0)#conta quanti ne escono per ogni porta ad ogni secondo
            density_near_doors[i] = []#densità intorno alla porta per ogni porta(come chiave)
            count_p[i] = 0


        ## Current time
        t = 0.0
        people, people_init_box_id, rng = people_initialization(N, init_people_box, dom,
                                                                dt, rmin, rmax, dmin=dmin,
                                                                seed=seed)

        vel_single_id = {}
        dist_percorsa = {}
        coord_old_single = {}
        k = 0
        if p_index != -1:
            for id in p_index:
                dist_percorsa[id] = 0
                vel_single_id[id] = []
                coord_old_single[id] = [people[k][0], people[k][1]]
                k += 1
        
        people_ids = people[:,3]
        for p in people:
            people_door_dist[p[3]] = [[distance.euclidean((p[0],p[1]), door_center)] for door_center in doors_center]
            people_velocities[p[3]] = [0]
            people_pos[p[3]] = [(p[0], p[1])]
        ## Array to store the results : all the people coordinates for all times
        Np = people.shape[0]
        Np_init = Np
        people_id = sp.arange(Np)
        results = sp.copy(people[:,:2]).reshape((Np,2,1))

        ## Array to store sensor data : time dir pts[2] for each sensor line
        if (len(sensors)>0):
            sensor_data = sp.zeros((Np,4,len(sensors)))

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
        vel_to_plot = []
        #print("Iterazione_" + str(it) + " di " + str(total))
        it += 1
        for p in people[:,3]:
            ts[p] = [0]
            people_velocities[p] = [0]

        while (t<Tf):
            contacts = compute_contacts(dom, people, dmax)
            I, J, Vd = compute_desired_velocity(dom, people)
            #print("===> time = ",t," number of persons = ",Np)

            if ((cc>=drawper) or (counter==0)):#per plot
                if plot:
                    print("plotting...")
                    plot_people(10, dom, people, contacts, Vd, people[:,2],
                                    plot_people=False, plot_contacts=plot_c,
                                    plot_velocities=plot_v, plot_paths=plot_pa,paths=results,
                                    plot_sensors=plot_s, sensors=sensors,
                                    savefig=True, filename=prefix+'fig_paths.png')

                    plot_people(10, dom, people, contacts, Vd, people[:,2], time=t,
                                    plot_people=plot_p, plot_contacts=plot_c,
                                    plot_velocities=plot_v, plot_paths=plot_pa,paths=results,
                                    plot_sensors=plot_s, sensors=sensors,
                                    savefig=True, filename=prefix+'fig_'+ \
                                    str(counter).zfill(6)+'.png')
                    #plot_grafici(20, ts, people_door_dist, t, people_velocities)
                    plt.pause(0.01)
                cc = 0
            info, B, U, L, P = projection(dt, people, contacts, Vd, dmin = dmin)

            if ferma_prof:
                #porta a destra
                if(Np > 1 and people[-1,0] >= doors_center[0][0]-1):
                    U[-1, :] = 0.0
                #porta a sinistra
                #if(Np > 1 and people[-1,0] <= doors_center[0][0]+1):
                    #U[-1, :] = 0.0
                #porta in alto
                #if(Np > 1 and people[-1,1] >= doors_center[0][1]-1):
                #    U[-1, :] = 0.0
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
                #people_velocities[p[3]].append(distance.euclidean(people_pos[p[3]], (p[0], p[1])))
                people_pos[p[3]].append((p[0], p[1]))
            t -= dt

            if len(U > 0):
                #instants.append(t)
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

                if p_index != -1:
                    for k in range(0, len(U)):
                        vel_single_id[p_index[k]].append(pow((U[k][0]*U[k][0]) + (U[k][1]*U[k][1]), 1/2))

                tot_vel.append(sum(velocities)/len(velocities))
            
            people, U, [people_id], id_exit = exit_door(2*dom.pixel_size, dom, people, U, doors_center, arrays=[ people_id])

            for id in id_exit:
                #aggiorno tempi e porta d'uscita
                time_single_unit[id].append(t)
                exit_single_unit[id][id_exit[id]] += 1

                #conto quanti ne escono per ogni porta
                count_exit_for_door[id_exit[id]] += 1
            Np = people.shape[0]

            ## Store people positions in the result array (used to draw people paths)
            tmp = 1e99*sp.ones((Np_init,2))
            tmp[people_id,:] = people[:,:2]
            results = sp.concatenate((results,tmp.reshape((Np_init,2,1))), axis=2)
            for id in id_exit:
                time[id] = t
            t += dt
            cc += 1
            counter += 1

            if p_index != -1:
                for k in range(0, len(people_id)):
                    dist = calculateDistance(coord_old_single[p_index[people_id[k]]][0], coord_old_single[p_index[people_id[k]]][1], people[k][0], people[k][1])
                    dist_percorsa[p_index[people_id[k]]] += dist
                    coord_old_single[p_index[people_id[k]]][0] = people[k][0]
                    coord_old_single[p_index[people_id[k]]][1] = people[k][1]

            if int_t < int(t):#scatta nuovo secondo
                #aggiorno il flow rate secondo per secondo
                flow = N_old - Np
                N_old = Np
                int_t = int(t)
                flows.append(flow)
                vel_to_plot.append(sum(velocities)/len(velocities))

                #calcolo la densità vicino alla porta
                count_custom = 0
                for p in people:
                    for a in area_near_doors:
                        area = area_near_doors[a]
                        #verifico se la persona è nell'area della porta
                        if p[0] > area[0][0] and p[0] < area[1][0]:#x
                            if p[1] > area[0][1] and p[1] < area[1][1]:#y
                                count_p[a] += 1
                    #verifico se la persona è nell'area a scelta custom_area
                    if p[0] > custom_area[0][0] and p[0] < custom_area[1][0]:#x
                        if p[1] > custom_area[0][1] and p[1] < custom_area[1][1]:#y
                            count_custom += 1   

                count_p_custom_area.append(count_custom)

                #salvo il flow rate per ogni porta e la densità vicino ad ogni porta
                for i in range(0, len(count_exit_for_door)):
                    flows_doors[i].append(count_exit_for_door[i])
                    density_near_doors[i].append(count_p[i] / area_near_doors_size[i])

                if debug: 
                    print("===> time = ",int_t," flow = ", flow, "flow per porta = ", count_exit_for_door, "number of persons = ", Np)
                    print(density_near_doors)
                #azzerro i contatore temporanei
                for i in range(0, len(door_lines)):
                    count_exit_for_door[i] = 0
                    count_p[i] = 0

            if (Np == 0):
                flow = N_old - Np
                flows.append(flow)
                for i in range(0, len(count_exit_for_door)):
                    flows_doors[i].append(count_exit_for_door[i])

                if debug: print("===> time = ",int_t," flow = ", flow, "flow per porta = ", count_exit_for_door, "number of persons = ", Np)

                for i in range(0, len(door_lines)):
                    count_exit_for_door[i] = 0
                if debug: print("END... Nobody here !")


                break

        #accellerazione
        #for v in velocities:
        #    print(len(instants))
        #    print(len(velocities[v]))
        #    plt.plot(instants, velocities[v])
        #plt.plot(range(0, len(tot_vel)), tot_vel)
        #plt.show()

        '''dist_single_unit = {}
        i = 1
        for v in vel_single_id:
            vel_single_id[v] = np.mean(vel_single_id[v])
            dist_single_unit[v] = vel_single_id[v] * time_single_unit[i][0]
            i += 1

        vel_real = []
        i = 0
        for dist in dist_percorsa:
            vel_real.append(dist_percorsa[dist] / times_real[i])
            i += 1'''

        flow_rate = sum(flows) / len(flows)
        flow_mean_total += flow_rate
        c_flows += 1
        mean = mean + t
        #f.write("velocity" + str(c_flows) + " = " + str(sum(tot_vel)/len(tot_vel)) + "m/s\n")
        #print("velocity" + str(c_flows) + " = " + str(sum(tot_vel)/len(tot_vel)) + "m/s")
        #print("evacuation time = ", t, "s")
        #print("path length = ", t * sum(tot_vel)/len(tot_vel), "m")
        '''f.write("velocità ogni istante: \n")
        f.write(str(vel_to_plot) + "\n")
        f.write("total_flows: \n")
        f.write(str(flows) + "\n")
        f.write("flows per ogni porta: \n")
        f.write(str(flows_doors) + "\n")
        f.write("densità vicino alle porte: \n")
        f.write(str(density_near_doors) + "\n")
        f.write("evacuation time_" + str(c_flows) +": " + str(t) + "\n")
        f.write("flow_rate_" + str(c_flows) + ": " + str(flow_rate) + "\n")
        f.write("persona nell'area a scelta: \n")
        f.write(str(count_p_custom_area) + "\n\n")'''
        times.append(t)

    flow_mean_total = flow_mean_total / c_flows
    mean = mean/ total
    f.write("flows per ogni porta:@")
    f.write(str(flows_doors) + "\n")
    f.write("all times\n")
    f.write(str(times))
    f.write("\n")
    f.write("mean ev: " + str(mean) + "\n")
    f.write("std ev: " + str(np.std(times)) + "\n")
    f.write("mean flow rates: " + str(flow_mean_total) + "\n")
    for id in time_single_unit:
        mean = sum(time_single_unit[id]) / len(time_single_unit[id])
        time_single_unit[id] = [mean]
        time_single_unit[id].extend(exit_single_unit[id])
        if len(door_lines) == 1:
            time_single_unit[id].append(0)

    f.write("time_single_unit:@")
    f.write(str(time_single_unit))
