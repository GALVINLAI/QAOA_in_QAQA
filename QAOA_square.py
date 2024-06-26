import numpy as np
import pennylane as qml
import math
from tqdm import tqdm
import json
from utilities import *
from QAOA import *


def qaoa_square(data_path:str, depth:int=1, sub_size:int=10):
    '''
    data_path : where graph data saved

    depth : depth level of QAOA circuit

    sub_size : allowable number of qubit

    return : 
    '''
    with open(data_path) as json_file:
        G = json.load(json_file)

    n_v = G['n_v']
    #d = G['d']
    edges = G['edges']
    # create a Graph
    G = Graph(v=list(range(n_v)), edges=edges)

    const = 0
    sols = {}
    # level indicate the hierarchy of QAOA
    level = 0 
    while G.n_v > sub_size:
        
        sols[level] = {}
        H = G.graph_partition(n=sub_size,policy='random')
        

        obj = []
        sol = []
        for H_sub in H:
            const_temp = 0.5 * sum([x[2] for x in H_sub.e])
            ret = qaoa(H_sub,const=const_temp,n_layers=depth,sample_method='max')
            obj.append(ret[0])
            sol.append(ret[1])
        sols[level]['sol'] = sol
        sols[level]['v'] = [h.v for h in H]
        n_sub = len(H)
        adjoint = np.zeros((n_sub,n_sub))
        for i in range(n_sub):
            for j in range(i+1, n_sub):
                w_pos = 0
                w_neg = 0
                for x in range(H[i].n_v):
                    for y in range(H[j].n_v):
                        m, n = H[i].v[x], H[j].v[y]
                        w_pos += (sol[i][x]!=sol[j][y]) * G.adj[m][n]
                        w_neg += (sol[i][x]==sol[j][y]) * G.adj[m][n]
                # w^prime
                adjoint[i][j]= w_neg-w_pos
                adjoint[j][i]= w_neg-w_pos
                const += w_pos
            const += obj[i]
        G = Graph(v=list(range(n_sub)), adjoint=adjoint)
        level += 1

    const_temp = 0.5 * sum([x[2] for x in G.e])
    ret = qaoa(G, const = const+const_temp, n_layers=depth)

    sols[level] = {}
    sols[level]['sol'] = ret[1]
    sols[level]['v'] = G.v

    return ret[0].item(), sols

if __name__ == '__main__':
    result = {}
    data_path = 'data/test.json'
    value, sols = qaoa_square(data_path=data_path, depth=1, sub_size=10)
    result['value'] = value
    result['sol'] = sols
    with open('result/test_result.json','w') as fp:
        json.dump(result,fp)

