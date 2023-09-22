

import math
import numba as nb
import heapq
import numpy as np

class Graph():
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj = dict()
        self.node_degrees = dict()
        self.sum_degrees = 0
        for i in range(self.num_nodes):
            self.adj[i] = set()
            self.node_degrees[i] = 0

class Edge():
    def __init__(self, i, j, weight):
        self.i = i
        self.j = j
        self.weight = weight

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.i != other.i:
                return False
            elif self.j != other.j:
                return False
            elif self.weight != other.weight:
                return False
            else:
                return True
        else:
            return False

    def __hash__(self):
        return hash((self.i,self.j,self.weight))

def read_graph(file_path):
    with open(file_path, 'r') as f:
        num_nodes = int(f.readline().strip())
        graph = Graph(num_nodes)
        for line in f.readlines():
            line = line.strip().split(' ')
            i = int(line[0])
            j = int(line[1])
            if i==j:
                continue
            weight = float(line[2])
            edge1 = Edge(i,j,weight)
            edge2 = Edge(j,i,weight)
            if not edge1 in graph.adj[i]:
                graph.adj[i].add(edge1)
                graph.adj[j].add(edge2)
                graph.node_degrees[i] += weight
                graph.node_degrees[j] += weight
                graph.sum_degrees += 2*weight
    return graph

def get_graph(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    num_nodes = A.shape[0]
    graph = Graph(num_nodes)
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] != A[j,i]:
                print("A[i,j] != A[j,i]")
            weight = (A[i,j]+A[j,i])/2
            if weight == 0:
                continue
            edge1 = Edge(i,j,weight)
            edge2 = Edge(j,i,weight)
            if not edge1 in graph.adj[i]:
                graph.adj[i].add(edge1)
                graph.adj[j].add(edge2)
                graph.node_degrees[i] += weight
                graph.node_degrees[j] += weight
                graph.sum_degrees += 2*weight
    return graph


@nb.jit(nopython=True)
def merge_deltaH(vi, vj, gi, gj, gx, sum_degrees):
    a1 = vi * np.log2(vi)
    a2 = vj * np.log2(vj)
    a3 = (vi + vj) * np.log2(vi + vj)
    a4 = gi * np.log2(vi / sum_degrees)
    a5 = gj * np.log2(vj / sum_degrees)
    a6 = gx * np.log2((vi + vj) / sum_degrees)
    return (a1+a2-a3-a4-a5+a6)/sum_degrees

class FlatSE():
    def __init__(self, A):
        graph = get_graph(A)
        self.graph = graph
        self.SE = 0
        self.communities = dict()
        self.pair_cuts = dict()
        self.connections = dict()
        for i in range(graph.num_nodes):
            self.connections[i] = set()

    def init_encoding_tree(self):
        for i in range(self.graph.num_nodes):
            if self.graph.node_degrees[i] == 0:
                continue
            ci = ({i}, self.graph.node_degrees[i], self.graph.node_degrees[i]) # nodes, volume, cut
            self.communities[i] = ci
            self.SE -= (self.graph.node_degrees[i] / self.graph.sum_degrees) * np.log2(self.graph.node_degrees[i] / self.graph.sum_degrees)
        for i in self.graph.adj.keys():
            for edge in self.graph.adj[i]:
                self.pair_cuts[frozenset([edge.i, edge.j])] = edge.weight
                self.connections[i].add(edge.j)
                self.connections[edge.j].add(i)

    def merge(self):
        merge_queue = []
        merge_map = dict()
        # merge_counter = itertools.count()
        for pair in self.pair_cuts.keys():
            commID1, commID2 = pair
            v1 = self.graph.node_degrees[commID1]
            v2 = self.graph.node_degrees[commID2]
            g1 = v1
            g2 = v2
            gx = g1 + g2 - 2 * self.pair_cuts[pair]
            deltaH = merge_deltaH(v1, v2, g1, g2, gx, self.graph.sum_degrees)
            # merge_count = next(merge_counter)
            merge_entry = [-deltaH, pair]
            heapq.heappush(merge_queue, merge_entry)
            merge_map[pair] = merge_entry

        while len(merge_queue) > 0:
            deltaH, pair = heapq.heappop(merge_queue)
            deltaH = -deltaH
            if pair == frozenset([]):
                continue
            if deltaH<0:
                continue
            commID1, commID2 = pair
            if (commID1 not in self.communities) or (commID2 not in self.communities):
                continue
            self.SE -= deltaH
            comm1 = self.communities.get(commID1)
            comm2 = self.communities.get(commID2)
            v1 = comm1[1]
            g1 = comm1[2]
            v2 = comm2[1]
            g2 = comm2[2]

            new_comm = (comm1[0].union(comm2[0]), v1+v2, g1+g2-2*self.pair_cuts[frozenset([commID1, commID2])])
            self.communities[commID1] = new_comm
            self.communities.pop(commID2)
            v1 = new_comm[1]
            g1 = new_comm[2]

            self.connections[commID1].remove(commID2)
            self.connections[commID2].remove(commID1)
            for k in self.connections[commID1]:
                if k in self.connections[commID2]:
                    pair_cut_1k = self.pair_cuts.get(frozenset([commID1, k])) + self.pair_cuts.get(frozenset([commID2, k]))
                    self.pair_cuts[frozenset([commID1, k])] = pair_cut_1k
                    self.connections[commID2].remove(k)
                    self.connections[k].remove(commID2)
                    self.pair_cuts.pop(frozenset([commID2, k]))
                    merge_entry = merge_map.pop(frozenset([commID2,k]))
                    merge_entry[-1] = frozenset([])
                else:
                    pair_cut_1k = self.pair_cuts[frozenset([commID1,k])]
                vk = self.communities[k][1]
                gk = self.communities[k][2]
                gx = g1 + gk - 2 * pair_cut_1k
                deltaH1k = merge_deltaH(v1, vk, g1, gk, gx, self.graph.sum_degrees)
                merge_entry = merge_map.pop(frozenset([commID1, k]))
                merge_entry[-1] = frozenset([])
                merge_entry = [-deltaH1k, frozenset([commID1, k])]
                heapq.heappush(merge_queue, merge_entry)
                merge_map[frozenset([commID1, k])] = merge_entry
            for k in self.connections[commID2]:
                pair_cut_2k = self.pair_cuts.get(frozenset([commID2, k]))
                vk = self.communities[k][1]
                gk = self.communities[k][2]
                gx = g1 + gk - 2 * pair_cut_2k
                deltaH1k = merge_deltaH(v1, vk, g1, gk, gx, self.graph.sum_degrees)
                self.pair_cuts[frozenset([commID1,k])] = pair_cut_2k
                self.pair_cuts.pop(frozenset([commID2,k]))
                merge_entry = merge_map.pop(frozenset([commID2,k]))
                merge_entry[-1] = frozenset([])
                merge_entry = [-deltaH1k, frozenset([commID1,k])]
                heapq.heappush(merge_queue, merge_entry)
                merge_map[frozenset([commID1,k])] = merge_entry
                self.connections.get(k).remove(commID2)
                self.connections.get(k).add(commID1)
                self.connections.get(commID1).add(k)
            self.connections.get(commID2).clear()

    def build_tree(self):
        self.init_encoding_tree()
        self.merge()
        y = self.to_label(self.communities)
        return y

    def to_label(self, communities):
        y_pred = np.zeros(self.graph.num_nodes, dtype=int)
        for i, ci in enumerate(communities):
            for vertex in communities[ci][0]:
                y_pred[vertex] = i
        return y_pred

if __name__=='__main__':
    graph = read_graph("E:/constrained_clustering/constrainedSE/lymph6graph/Lymph6Graph")
    flatSE = FlatSE(graph)
    flatSE.build_tree()
    print(flatSE.communities)
    print(flatSE.SE)




