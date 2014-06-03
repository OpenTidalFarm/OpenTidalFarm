# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:01:35 2013

@author: dmc13
"""

from matplotlib import pyplot as plt
import math
import copy
import numpy as np
from ad import adnumber
import ad
from ad.admath import *
import random
from helpers import info, info_red, info_green, info_blue, get_rank
import dolfin


class CableCostGA(object):
    
    def __init__(self, substation_location = [[0,0]], capacity = 8, pop_size = 8000, num_iter = 2200, convergence_definition = 21, scaling_factor = 3900, show_prog = False, show_result = False, redundancy = False, basic_mode = False): 
        self.substation_location = substation_location
        self.capacity = capacity
        self.pop_size = pop_size
        self.num_iter = num_iter
        self.convergence_definition = convergence_definition
        self.scaling_factor = scaling_factor
        self.show_prog = show_prog
        self.show_result = show_result
        self.length_record = []
        self.redundancy = redundancy
        if len(substation_location) == 1:
            self.multiple_subs = False
        else: self.multiple_subs = True
        self.basic_mode = basic_mode
    
    
    def cable_info(self):
        rank = get_rank()
        if rank == 0:
            print '\n=== Cable routing parameters ==='
            print 'Substation location(s): ', self.substation_location
            print 'Cable capacity: ', self.capacity
            print 'Population size: ', self.pop_size
            print 'Maximum number of iterations: ', self.num_iter
            print 'Convergence definition: ', self.convergence_definition
            print 'Scaling factor: ', self.scaling_factor
            print 'Redundancy: ', self.redundancy
            if self.basic_mode:
                print 'OPERATING IN BASIC ROUTING MODE: ALL TURBINES CONNECTED DIRECTLY TO SUBSTATION'
            print '\n'

    def convert_to_adnumber(self, coordinate_list):
        '''Convert the location vectors from floats into adnumbers to enable differentiation'''
        adnumber_coordinate_list = []
        for i in range(len(coordinate_list)):
            adnumber_coordinate_list.append([adnumber(coordinate_list[i][0]), adnumber(coordinate_list[i][1])])
        coordinate_list = adnumber_coordinate_list
        return coordinate_list
        
    
    def differentiate(self, best_chromosome, turbine_locations):
        '''Differentiate the length of the routing w.r.t. the position of the turbines, produces a n x 2 array, ((dC/dx1, dC/dy1), ...)'''
        print 'Determining dC/dX: Performing automatic differentiation...'
        turbine_locations = self.convert_to_adnumber(turbine_locations)
        substation_location = self.convert_to_adnumber(self.substation_location)
        vertices = substation_location + turbine_locations
        C = self.construct_cost_matrix(vertices)
        route = best_chromosome[0]
        breaks = best_chromosome[1]    
        rting = self.routing(route, breaks)
        total_dist = self.routing_distance(rting, C)
        n = len(turbine_locations)
        dC_dX = np.zeros((n,2)) 
        for i in range(n):
            dC_dX[i][0] = total_dist.d(turbine_locations[i][0])
            dC_dX[i][1] = total_dist.d(turbine_locations[i][1])
        print 'Automatic differentiation complete...'
        return dC_dX
    
    
    def construct_cost_matrix(self, vertices):
        '''Constructs a matrix of costs for every potential edge - connections of vertices to themselves are set to infinity'''
        distances = []
        grouped_distances = []
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i == j:
                    dist = 99
                else:
                    dist = (sqrt((vertices[i][0] - vertices[j][0])**2 + (vertices[i][1] - vertices[j][1])**2))
                distances.append(dist)
        for i in range(0, len(distances), len(vertices)):
            grouped_distances.append(tuple(distances[i:i+len(vertices)]))
        C = np.array(grouped_distances)
        return C
    
    
    def produce_plot(self, vertices, best_chromosome, global_min):
        '''Display with matplotlib'''
        opt_route = best_chromosome[0]
        opt_breaks = best_chromosome[1]
        R = self.routing_coordinates(vertices, self.routing(opt_route, opt_breaks))
        plt.title('Total distance='+str(global_min))
        V = np.array((vertices))
        plt.plot(V[:,0],V[:,1],'.')
        for i in range(len(R)):
            plt.plot(R[i][:,0], R[i][:,1], '-')
#        for i in range(len(V)):
#            plt.text(V[i][0], V[i][1], '%s' % (str(i)))
        basin_x = 640.
        basin_y = 2560.
        site_x = 320.
        site_y = 1280.
        site_x_start = ((basin_x - site_x)/2)-230
        site_y_start = (basin_y - site_y)/2
        #plt.plot([site_x_start, site_x_start + site_x, site_x_start + site_x, site_x_start, site_x_start], [site_y_start, site_y_start, site_y_start + site_y, site_y_start + site_y, site_y_start], linestyle = '--', color = 'r')
        plt.axis('equal')
        plt.savefig('destination_path02.pdf', format='pdf', dpi=1000)
        plt.show()
        
    
    def rand_breaks(self, n, min_route, n_routes, n_breaks):
        '''Produces a list of random, but valid, route-breaks'''
        RB = [np.random.random_integers(min_route, self.capacity)]
        for i in range(1, n_breaks):
            RB.append(np.random.random_integers(min_route, self.capacity) + RB[i-1])
        if RB[-1] < (n - self.capacity):
            short = (n - self.capacity) - RB[-1]
            add_each = int(np.ceil(0.5 + short / len(RB)))
            for i in range(len(RB)):
                RB[i] = RB[i] + add_each * (i+1)
        if self.multiple_subs:
#            print 'hello', RB
            RB = [0] + RB
            for i in range(len(RB)):
                RB[i] = [RB[i], random.randint(0,len(self.substation_location)-1)]
        return RB    
    
        
    def initialise_population(self, n, min_route, n_routes, n_breaks, prev_routing):
        '''Randomly produces an initial population'''
        popbreaks = [prev_routing[1]]
        poproutes = [prev_routing[0]]
        for i in range(self.pop_size - 1):
            popbreaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))
            poproutes.append(random.sample(range(len(self.substation_location), n+len(self.substation_location)), n))
#        print poproutes
        return poproutes, popbreaks
    
    
    def routing(self, route, breaks):
        '''Combine route and breaks into an array of the routes described as vertices in the order in which they are toured'''  
        if self.redundancy:
            rting = [[0] + route[0:breaks[0]] + [0]]
            if len(breaks) > 1:
                for f in range(1, len(breaks)):
                    rting.append([0] + route[breaks[f-1]:breaks[f]] + [0])
            rting.append([0] + route[breaks[-1]:] + [0])
            return rting
        elif self.multiple_subs:
#            print route, breaks
            rting = []#[[breaks[0][1]] + route[0:breaks[1][0]]]
            #print breaks
            #if len(breaks) > 1:
            for f in range(len(breaks)-1):
                rting.append([breaks[f][1]] + route[breaks[f][0]:breaks[f+1][0]])
            rting.append([breaks[-1][1]] + route[breaks[-1][0]:])
#            print 'rting', rting
            return rting
        else:    
            rting = [[0] + route[0:breaks[0]]]
            if len(breaks) > 1:
                for f in range(1, len(breaks)):
                    rting.append([0] + route[breaks[f-1]:breaks[f]])
            rting.append([0] + route[breaks[-1]:])
            return rting
    
    
    def routing_distance(self, routing, C):
        '''Return the geometric length of the routing'''
        d = 0
        for r in range(len(routing)):
            for v in range(len(routing[r]) - 1):
                d += C[routing[r][v]][routing[r][v+1]]
        return d
    
    
    def routing_coordinates(self, vertices, rting):
        '''Convert a routing expressed in indexes to a routing expressed in coordinates''' 
        for i in range(len(rting)):
            for j in range(len(rting[i])):
                rting[i][j] = vertices[rting[i][j]]
            rting[i] = np.array(rting[i])
        return rting
    
    
    def evaluate_population(self, pop, C, n):
        '''produce a list of the functional evaluations of each member of the population'''
        D = []
        for i in range(self.pop_size):
            D.append(self.routing_distance(self.routing(pop[0][i], pop[1][i]), C))
        return D
        
        
    def clarke_wright(self, route, breaks, vertices, turbine_locations):
        '''apply clarke and wright algorithm to chromosome'''
        rting = self.routing(route, breaks)
        rting_coords = self.routing_coordinates(vertices, rting)      
        rting = []
        for i in range(len(rting_coords)):
            temp_r = rting_coords[i].tolist()
            del temp_r[0]
            CW = Clarke_Wright(temp_r, self.substation_location)
            temp_r = CW.run()
            del temp_r[0]
            for j in range(len(temp_r)):
                if temp_r[j] not in turbine_locations:
                    return range(1, len(turbine_locations)+1)
                else:
                    temp_r[j] = turbine_locations.index(temp_r[j]) + 1
            rting += temp_r
        return rting
    
    
    def transformations(self, bestof8route, bestof8breaks, n, min_route, n_routes, n_breaks, vertices, iteration, turbine_locations):
        '''Returns a copy of the original chromosome and 7 mutations'''
        selector = random.sample(range(len(self.substation_location), n), n-len(self.substation_location))
        randlist = [selector[n/3], selector[2*n/3]]
        I = min(randlist)
        J = max(randlist)
        temp_pop_route = []
        temp_pop_breaks = []
        temp_pop_route.append(bestof8route)
        temp_pop_breaks.append(bestof8breaks)
        # transformation 1  
        trans_1 = copy.deepcopy(bestof8route)
        trans_1[I], trans_1[J] = trans_1[J], trans_1[I]
        temp_pop_route.append(trans_1)
        temp_pop_breaks.append(bestof8breaks)
        ## transformation 2
        trans_2 = copy.deepcopy(bestof8route)
        Temp = trans_2[I:J]; Temp = Temp[::-1]; trans_2 = trans_2[:I] + Temp + trans_2[J:]
        temp_pop_route.append(trans_2)
        temp_pop_breaks.append(bestof8breaks)
        ## transformation 3
        trans_3 = copy.deepcopy(bestof8route)
        trans_3.remove(I); trans_3.insert(J, I)
        temp_pop_route.append(trans_3)
        temp_pop_breaks.append(bestof8breaks)
        ## transformation 4
        if (iteration%10) == 0:
#        if iteration > 15:
            #print 'Transformation 4 substituted with Clarke and Wright'
            trans_4 = copy.deepcopy(bestof8route)
            trans_4 = self.clarke_wright(trans_4, bestof8breaks, vertices, turbine_locations)
            valid = True
            for i in range(1, len(trans_4)+1):
                if i not in trans_4:
                    valid = False
            if valid:
                temp_pop_route.append(trans_4)
                temp_pop_breaks.append(bestof8breaks)
            else:
                temp_pop_route.append(bestof8route)
                temp_pop_breaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))
        else:
            temp_pop_route.append(bestof8route)
            temp_pop_breaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))
        ## transformation 5
        temp_pop_route.append(trans_1)
        temp_pop_breaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))
        ## transformation 6        
        temp_pop_route.append(trans_2)
        temp_pop_breaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))
        ## transformation 7        
        temp_pop_route.append(trans_3)
        temp_pop_breaks.append(self.rand_breaks(n, min_route, n_routes, n_breaks))        
        
        return temp_pop_route, temp_pop_breaks
        
     
    def breed_population(self, pop, D, n, min_route, n_routes, n_breaks, vertices, iteration, turbine_locations):
        '''Create a new population based upon the best performing specimens of the old population'''
        n = len(turbine_locations)        
        chrome_pop = []
        for i in range(self.pop_size):
            chrome_pop.append(pop[0][i] + pop[1][i] + [D[i]])
        random.shuffle(chrome_pop)
        new_pop_route = []
        new_pop_breaks = []
        for i in range(0, self.pop_size, 8):
            Dists = [] 
            for j in range(8):
                Dists.append(chrome_pop[i+j][-1])
            MIndx = min(xrange(len(Dists)),key=Dists.__getitem__)
            bestof8Chrome = chrome_pop[MIndx+i]
            bestof8route = bestof8Chrome[:n]
            bestof8breaks = bestof8Chrome[n:-1]
            new_8 = self.transformations(bestof8route, bestof8breaks, n, min_route, n_routes, n_breaks, vertices, iteration, turbine_locations)
            new_pop_route += new_8[0]
            new_pop_breaks += new_8[1]
        return new_pop_route, new_pop_breaks
 
   
    def graph_iterations(self):
        y = np.array(self.length_record)
        x = np.array(range(len(y)))
        plt.plot(x,y)
        plt.show()
        
        
    def basic_connection_mode(self, vertices, turbine_locations):
        chromosome = [range(1,len(turbine_locations)+1), range(1,len(turbine_locations))]
        C = self.construct_cost_matrix(vertices)
        length = self.routing_distance(self.routing(chromosome[0], chromosome[1]), C)
        if self.show_result:
            self.produce_plot(vertices, chromosome, length)
        return chromosome, length
            
            
    def run_GA(self, turbine_locations, prev_routing):
        '''Run the genetic algorithm'''
        np.random.seed(100)
        random.seed(100)        
        vertices = self.substation_location + turbine_locations
        
        if self.basic_mode:
            print 'In basic mode: all turbines connected directly to substation'
            return self.basic_connection_mode(vertices, turbine_locations)
        
        n_routes = int(math.ceil(float(len(vertices)) / self.capacity))
        min_route = len(vertices) / n_routes
        n = len(turbine_locations)
        n_breaks = n_routes - 1
        converged = False
        convergence_counter = 0
        print 'Running Genetic Algorithm...'
        pop = self.initialise_population(n, min_route, n_routes, n_breaks, prev_routing)
        C = self.construct_cost_matrix(vertices)
        D = self.evaluate_population(pop, C, n)    
        MIndx = min(xrange(len(D)),key=D.__getitem__)    
        global_min = copy.deepcopy(D[MIndx])
        best_chromosome = [copy.deepcopy(pop[0][MIndx]), copy.deepcopy(pop[1][MIndx])]
        dist_history = [global_min]
    
        for iteration in range(self.num_iter):
            if len(self.length_record) > (self.convergence_definition + 5) and (self.length_record[-(self.convergence_definition + 2)] / self.length_record[-1]) < 1.01:
                converged = True
            if not converged:
                if self.show_prog:
                    self.length_record.append(global_min)
                    print 'Current Routing Length %2f' % global_min
                pop = self.breed_population(pop, D, n, min_route, n_routes, n_breaks, vertices, iteration, turbine_locations)
                D = self.evaluate_population(pop, C, n)
                MIndx = min(xrange(len(D)),key=D.__getitem__)
                if D[MIndx] < global_min:
                    global_min = copy.deepcopy(D[MIndx])
                    best_chromosome = [copy.deepcopy(pop[0][MIndx]), copy.deepcopy(pop[1][MIndx])]
                    dist_history.append(global_min)
                    convergence_counter = 0
                else: convergence_counter += 1
                if convergence_counter > self.convergence_definition:
                    converged = True
        if self.show_result:
            self.produce_plot(vertices, best_chromosome, global_min)
        print 'GA Complete, Routing Length is: %2f' % global_min
        #self.graph_iterations()
        return best_chromosome, global_min
        
    def compute_cable_cost(self, turbine_locations, prev_routing):
        if self.show_prog: self.cable_info()
        if prev_routing == None:
            dummy_route = random.sample(range(len(self.substation_location), len(turbine_locations)+len(self.substation_location)), len(turbine_locations))
            dummy_break = range(self.capacity, len(turbine_locations), self.capacity)
            if self.multiple_subs:
                dummy_break = [0] + dummy_break
                for i in range(len(dummy_break)):
                    dummy_break[i] = [dummy_break[i], random.randint(0,len(self.substation_location)-1)]
            prev_routing = [dummy_route, dummy_break]
        print 'Previous Routing: %s' % (str(prev_routing))
        out = self.run_GA(turbine_locations, prev_routing)
        self.best_chromosome = out[0]
        length = out[1]*self.scaling_factor
        print 'Scaled cable length = %2f' % (length)
        return length, self.best_chromosome
        
    def compute_cable_cost_derivative(self, turbine_locations):
        dC_dX = self.differentiate(self.best_chromosome,
                turbine_locations)*self.scaling_factor
        return dC_dX
        
        for i in range(len(RB)):
                RB[i] = [RB[i], random.randint(0,len(self.substation_location)-1)]
        
        
## 0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0 ##




class Clarke_Wright(object):
    
    def __init__(self, turbine_locations, substation_location, plot_output = False):
        self.turbine_locations = turbine_locations
        self.substation_location = substation_location
        self.plot_output = plot_output
      
    def routing_coordinates(self, vertices, rting):
        '''Convert a routing expressed in indexes to a routing expressed in coordinates'''        
        for i in range(len(rting)):
            rting[i] = vertices[rting[i]]
        return rting
    

    def routing_indexes(self, R, vertices):
        '''Convert a routing expressed in tuples to a routing expressed in indexes'''
        chromosome = [0]
        while len(chromosome) <= len(R):
            for i in range(len(R)):
                if R[i][0] == chromosome[-1]:
                    chromosome.append(R[i][1])
                if R[i][1] == chromosome[-1]:
                    chromosome.append(R[i][0])
        return chromosome
    
    
    def produce_plot(self, R, vertices):
        '''Display with matplotlib'''
        R = self.routing_indexes(R, vertices)
        R = self.routing_coordinates(vertices, R)
        R = np.array(([R])) 
        V = np.array((vertices))
        plt.plot(V[:,0],V[:,1],'o')
        plt.plot(R[0][:,0], R[0][:,1], '-')
        plt.text(V[0][0], V[0][1], '%s' % (str(0)))
        plt.axis('equal')
        plt.show() 
        

    def construct_cost_matrix(self, vertices):
        '''Constructs a matrix of costs for every potential edge - connections of vertices to themselves are set to infinity'''
        distances = []
        grouped_distances = []
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i == j:
                    dist = 99
                else:
                    dist = (sqrt((vertices[i][0] - vertices[j][0])**2 + (vertices[i][1] - vertices[j][1])**2))
                distances.append(dist)
        for i in range(0, len(distances), len(vertices)):
            grouped_distances.append(tuple(distances[i:i+len(vertices)]))
        C = np.array(grouped_distances)
        return C
        

    def make_graph(self, Rgraph, no_depot = True):
        '''Produces a dictionary representing the graph of the form (vertex: [connected vertices], ...) can remove the depot so as to seperate the routes from one another'''
        if no_depot:
            remove_list=[]
            for edge in Rgraph:
                if edge[0] == 0 or edge[1] == 0:
                    remove_list.append(edge)
            for edge in remove_list: 
                Rgraph.remove(edge)
        vertices = []
        for a_tuple in Rgraph:
            vertices.extend(list(a_tuple))
        vertices = list(set(vertices))
        nVertices = len(vertices)
        G = {}
        for i in range(0,nVertices):
            G[vertices[i]]=[]
        for edge in Rgraph:
            G[edge[1]].append(edge[0])
            G[edge[0]].append(edge[1])
        for vertex in G:
            G[vertex] = list(set(G[vertex]))
        return G
        

    def initialise_R(self, n):
        '''produce the initial edge set - all clients connected to their nearest depot'''
        R = []
        # For each client vertex, identify the nearest depot and add (v,d) to initial edge list R
        for index in range(1, n):
            R.append((index, 0))
        return R
        

    def initialise_G(self, R, n):
        '''produce the graph of the initial edge set'''
        G = []
        Rgraph = copy.deepcopy(R)
        G = self.make_graph(Rgraph)
        return G
        

    def construct_savings_list(self, C, n):
        '''Constructs a list of the savings in making every connection and orders them'''
        S = []    
        # For each possible pair of clients in the graph, calculate the cost saving from joining with edge, and add to list S
        R = self.initialise_R(n)        
        for vertex1 in range(1, n):
            for vertex2 in range(1, n):
                if not vertex1 == vertex2:
                    depot = [j[1] for i, j in enumerate(R) if j[0] == vertex1]
                    saving = C[vertex1, depot] - C[vertex1, vertex2]
                    S.append((vertex1, vertex2, saving))
        # Sort the list S of cost savings in decreasing order
        S.sort(key = lambda s:s[2], reverse = True)
        return S


    def find_path_length(self, graph, start, end, path=[]):
        '''Returns a list of vertices on a path in order that they are connected'''        
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        for node in graph[start]:
            if node not in path:
                newpath = self.find_path_length(graph, node, end, path)
                if newpath: return newpath
        return None


    def neighbour_depot(self, edge, R, Vd):
        '''Ensures that k is neighbouring a depot - ensures no short-circuiting'''      
        nhbrDepot = False
        if (edge[0], 0) or (0, edge[0]) in R:
            nhbrDepot = True
        return nhbrDepot

        
    def one_neighbour(self, edge, R):
        '''Prevent route branching by checking that u has only one neighbour'''
        oneNhbr = True
        for edge_temp in R:
            if edge_temp[1] != 0 and edge_temp[0] == edge[0]:
                oneNhbr = False
            if edge_temp[1] == edge [1]:
                oneNhbr = False
        return oneNhbr

    
    def perform_checks(self, S, G, R, n, edge):
        '''Performs viability checks on proposed saving'''
        if not (self.find_path_length(G, edge[0], edge[1])==None):           
            return False
        if not self.neighbour_depot(edge, R, range(1, n)):
            return False
        if not self.one_neighbour(edge, R):
            return False
        return True

        
    def k_d(self, edge, R):
        '''Defines the edge whose removal is proposed'''
        if (edge[0],0) in R:
            k_d = (edge[0],0)
        elif (0,edge[0]) in R:
            k_d = (0,edge[0])
        return k_d

        
    def perform_merge(self, edge, R, k_d):
        '''Removes the superfluous old edge and inserts the new edge & updates the routing graph'''
        R.remove(k_d)
        R.append((edge[0], edge[1]))
        Rgraph = copy.deepcopy(R)
        G = self.make_graph(Rgraph)
        return G

        
    def run(self):
        vertices = self.substation_location + self.turbine_locations
        n = len(self.turbine_locations)+1
        C = self.construct_cost_matrix(vertices)
        S = self.construct_savings_list(C, n)
        R = self.initialise_R(n)
        G = self.initialise_G(R, n)
        while (not S == []) and (S[0][2][0] > 0):
            edge = (S[0][0],S[0][1])
            allowable = self.perform_checks(S, G, R, n, edge)
            if allowable:
                G = self.perform_merge(edge, R, self.k_d(edge, R))
            else:                 
                del S[0]
        if self.plot_output:
            self.produce_plot(R, vertices)
        R = self.routing_indexes(R, vertices)
        R = self.routing_coordinates(vertices, R)
        return R

        
        
if __name__ == '__main__':

    turbine_locations = [[74, 101], [58, 41], [72, 67], [116, 86], [97, 103], [104, 70], [10, 96], [55, 14], [86, 20], [52, 36], [49, 117], [118, 127], [65, 59], [17, 5], [23, 74], [40, 35], [82, 49], [64, 50], [79, 83], [59, 95], [29, 107], [41, 80], [124, 18], [103, 42], [7, 66], [43, 68], [60, 8], [36, 17], [51, 89], [125, 84], [123, 54], [85, 100], [109, 118], [84, 47], [0, 19], [117, 63], [3, 51], [96, 1], [89, 109], [76, 28], [111, 30], [62, 104], [5, 0], [70, 13], [35, 11], [87, 26], [14, 60], [102, 81], [48, 29], [121, 91], [13, 53], [78, 31], [2, 37], [127, 122], [69, 3], [113, 93], [15, 121], [67, 62], [1, 72], [95, 57], [12, 55], [63, 38], [105, 115], [126, 23], [9, 88], [101, 24], [4, 2], [45, 69], [114, 112], [32, 40], [77, 123], [30, 43], [93, 27], [100, 124], [75, 45], [20, 56], [119, 16], [6, 22], [83, 98], [98, 79], [21, 78], [26, 61], [112, 125], [110, 10], [115, 85], [24, 73], [8, 116], [44, 25], [94, 113], [108, 92], [34, 126], [57, 52], [99, 97], [92, 44], [90, 111], [25, 114], [50, 48], [68, 94], [27, 33], [91, 82], [106, 90], [88, 9], [16, 120], [61, 4], [31, 15], [53, 65], [22, 75], [73, 106], [107, 34], [47, 108], [71, 58], [122, 99], [11, 76], [80, 64], [19, 12], [37, 39], [81, 110], [38, 71], [42, 77], [28, 119], [18, 32], [46, 105], [56, 7], [54, 87], [39, 46], [120, 6], [66, 102], [33, 21]]
    turbine_locations = [[343.5000000000009, 1109.9999999999432], [449.56193055171383, 1204.9568212253362], [373.8701934484227, 1372.521503004281], [363.63561883807995, 1562.6649907636972], [343.4999999999962, 1690.0000000000691], [533.0506158875066, 1110.1311130904792], [452.18317292830216, 1234.842086646732], [373.5269830186765, 1414.7654917011073], [537.9442839316735, 1549.897917515539], [540.112139613262, 1610.1238083439873], [532.8823929868996, 1149.9040738644953], [598.5760252943064, 1280.0962562922723], [553.167508585214, 1400.12916708796], [535.3656776324977, 1470.3482155216236], [535.1333650495839, 1654.9090490810224], [595.6766848546965, 1250.1831082868339], [623.4872156891772, 1329.8702009501885], [545.0485125737644, 1441.9538022061836], [537.8677401792739, 1519.8980151615397], [530.3391422667778, 1684.8940274896208]]
    show_prog = True
    print len(turbine_locations)
    CC = CableCostGA(show_prog = True, show_result = True)
    
    timer = dolfin.Timer("Cable Length Evaluation")
    prev_routing = None#[[5, 2, 6, 7, 3, 4, 8, 1], [7]]
    output = CC.compute_cable_cost(turbine_locations, prev_routing = None)
    timer.stop()

    print output[0], output[1]
    print 'Cable length evaluation in: ', timer.value(), 'seconds'

    timer = dolfin.Timer("Cable Length Evaluation")
    deriv = CC.compute_cable_cost_derivative(turbine_locations)
    timer.stop()

    print deriv
    print 'Derivative evaluation in: ', timer.value(), 'seconds'
    






