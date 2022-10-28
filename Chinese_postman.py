import itertools
import logging
import os,sys
import threading
from time import sleep
from Graph import Graph
from collections import defaultdict, OrderedDict
import heapq as heap
from copy import deepcopy
from tqdm import tqdm
import math



def sum_edges(G):
    ''' Function that return the weigth sum of all edge in the input's graph'''

    return G.cost

def dijkstra(G, startingNode, finishNode):
    '''
    Function that implement the Dijkstra's algorithm to find the shortest paths between the startingNode and the finishNode (in input)
    Return: a dictionary “parentsMap” to reconstruct the path after the execution of the algorithm,
            a dictionary “nodeCosts” for keeping track of minimum costs for reaching different nodes from the source node.
    '''

    visited = set()
    parentsMap = {}
    pq = []
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[startingNode] = 0
    heap.heappush(pq, (0, startingNode))
 
    while pq:
    
		# go greedily by always extending the shorter cost nodes first
        _, node = heap.heappop(pq)
        #print(f"nodo pop {node}")
        if(node == finishNode): 
            break

        visited.add(node)
 
        for adjNode, weight in G.graph[node]:
            if adjNode in visited:	continue
			
            #print(f"adj list {adjNode}")	
            newCost = nodeCosts[node] + weight
            if nodeCosts[adjNode] > newCost:
                parentsMap[adjNode] = (node,weight)
                nodeCosts[adjNode] = newCost
                heap.heappush(pq, (newCost, adjNode)) 
       
    return parentsMap, nodeCosts        



def get_odd(G):
    ''' Function to find odd degree vertices in input's graph '''

    odds = []
    for node, adjlist in G.graph.items():
       if(len(adjlist)%2 != 0):
           odds.append(node)
              
    return odds

''''
def gen_pairs(odds):
     Function to generate all possible pairs 

    pairs = []

    for pair in itertools.combinations(odds,2):
        pairs.append(pair)

    #print('pairs are:',pairs)
    return pairs

def get_ppairs(pairs,odds):
     Function to generate unique pairs 

    pa = []
    print(pairs)
    totale = int(math.factorial(len(pairs)) / (math.factorial(int(len(odds)/2)) * math.factorial(len(pairs)-int(len(odds)/2))))
    print(totale)
    with tqdm(total=totale) as pbar:
        for p in itertools.combinations(pairs, int(len(odds)/2)):
            pbar.update(1)
            pa.append(p)
            for h in range(0, int(len(odds)/2)-1):
                for i in range(2):
                    for j in range(h+1,int(len(odds)/2)):
                        for k in range(2):
                            if(p[h][i] == p[j][k]):
                                try:
                                    pa.remove(p)
                                except ValueError:
                                    pass
    return pa
'''

def pairings(remainder, partial = None):
    ''' Function to generate unique pairs '''

    partial = partial or []

    if len(remainder) == 0:
        yield partial

    else:
        for i in range(1, len(remainder)):
            pair = [(remainder[0], remainder[i])]
            r1   = remainder[1:i]
            r2   = remainder[i+1:]
            for p in pairings(r1 + r2, partial + pair):
                yield p

def calculate_npairings(n):
    ''' Function that return the total number of all possible unique pairs given n odd nodes '''

    if n == 1:
        return n
    else:
        return n * calculate_npairings(n-2)


def get_ppair(odds):
  
    pairr = list()

    total = calculate_npairings(len(odds)-1)
    with tqdm(total=total) as pbar:
        for p in pairings(odds):
            pbar.update(1)
            pairr.append(tuple(p))
    
    return pairr


def backtrack(pm,i,j,lista = None):
    ''' Function to reconstruct the path from node i to node j after the execution of dijkstra's algorithm '''

    if lista is None:
        lista = []
    if j == i:
        return lista
    else:
        lista.append((j,pm[j][0],pm[j][1]))
        backtrack(pm,i,pm[j][0],lista)
    return lista
    

def sp_unique_pairs(graph,pairings_sum,index):
    """ Thread function that execute dijkstra for all the pairs of all possible perfect matching
        and store the one whose cost is minimum """

    key = index + 1
    pp_costs[key] = []
    partial_min_cost = 1000000000
    prova = list()
    sleep(0.1) #without tqdm is glitched
    for i in tqdm(pairings_sum):
        index+=1
        #print(f"Perfect pair:  {i}\n")
        cost = 0

        for j in range(len(i)):
            pm , dijkk = dijkstra(graph, i[j][0], i[j][1])
            dijk = dijkk[i[j][1]] #prendo il costo del cammino minore per arrivare dal nodo i al nodo j
            #print(f"pm {pm} nodecost {dijkk}")
            #print(backtrack(pm,i[j][0],i[j][1]))       
            prova.append((pm,i[j])) #store the parentsmap and the pair of a perfect matching
            cost += dijk
            #print(f"cost:  {s}")
        if cost < partial_min_cost: 
            partial_min_cost = cost
            partial_min_pair = deepcopy(prova)
        else:
            prova.clear()

    pp_costs[key] = partial_min_pair
    pp_costs[key].append(partial_min_cost)

    


#Chinese_Postman implementation
def chinese_postman(graph):
    ''' 
        Function that execute the chinese postman algorithm 
        and return the computed distance and the modified graph to which edges of the minimum cost perfect pair have been added. 
    '''

    sum_edge = sum_edges(graph)
    odds = get_odd(graph)
    print(f"N° of odd nodes: {len(odds)}\n")
    if(len(odds)==0):
        return sum_edge
 
    #pairs = gen_pairs(odds)
    #pairings_sum = get_ppairs(pairs,odds)
    print(f"Generate all the unique pairs:")
    pairings_sum = get_ppair(odds)
   

    total = calculate_npairings(len(odds)-1)
    print(f"\nCompute dijkstra for all the {total} pairs:")
    global pp_costs   
    pp_costs = {}

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    
    threads = list()
    for index in range(2):
        logging.info("CPP : create and start thread %d.", index)
        if index == 0:
            thread = threading.Thread(target=sp_unique_pairs, args=(graph,pairings_sum[(total//2):],total//2-1))
            threads.append(thread)
            thread.start()
        else:
            thread = threading.Thread(target=sp_unique_pairs, args=(graph,pairings_sum[:(total//2)],-1))
            threads.append(thread)
            thread.start() 

    for index, thread in enumerate(threads):
        logging.info("CPP : before joining thread %d.", index)
        thread.join()
        logging.info("CPP : thread %d done", index)
    

    # After threads execution sort the output dict by cost
    ordered_dict = dict(sorted(pp_costs.items(), key=lambda item: item[1][len(item[1])-1]))
    min_ppair = list(ordered_dict.items())[0][1] #take the first element of the sorted dict (minimum cost)
    #print(min_ppair)

    #Adding the edges of the minimum cost perfect pair to make the graph an Elurian one
    modified_graph = deepcopy(graph)
    for elem in min_ppair[:-1]:
        pm = elem[0]
        edges_toadd = backtrack(pm,elem[1][0], elem[1][1])
       
        for elem in edges_toadd:
            modified_graph.addEdge(elem[0],elem[1],elem[2])
       
   
           
    #take the weight of the min_ppair
    added_dis = min_ppair[len(list(ordered_dict.items())[0][1])-1]
    print(f"\nOriginal edge sum: {sum_edge} Added distance: {added_dis}\n")
    #compute the chinese postman distance 
    chinese_dis = sum_edges(modified_graph)  #or chinese_dis = sum_edge + added_dis

    return chinese_dis, modified_graph  
    


