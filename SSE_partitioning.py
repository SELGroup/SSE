import numba as nb
import heapq
import numpy as np

from SE_partitioning import Graph, Edge, read_graph, get_graph

EPS = 1e-15

@nb.jit(nopython=True)
def cal_module_SSE(g, sum_degrees, v, v_minus):
    return -(g/sum_degrees) * np.log2(v/v_minus)

class FlatSSE():
    def __init__(self, A, A_con, num_cluster=None, mustlink_first = None):
        self.mustlink_first = mustlink_first
        self.num_cluster = num_cluster
        A = A - np.diag(np.diag(A))
        A_con = A_con - np.diag(np.diag(A_con))
        self.A = A
        self.A_con = A_con
        self.graph = get_graph(A)
        self.graph_con = get_graph(A_con)
        self.SSE = 0
        self.communities = dict()
        self.pair_cuts = dict()
        self.pair_cuts_con = dict()
        self.connections = dict()
        self.connections_con = dict()
        for i in range(self.graph.num_nodes):
            self.connections[i] = set()
            self.connections_con[i] = set()

    def init_encoding_tree(self):
        for i in range(self.graph.num_nodes):
            if self.graph.node_degrees[i] == 0:
                continue
            SSEi = - ((self.graph.node_degrees[i]) / self.graph.sum_degrees) * np.log2(
                self.graph.node_degrees[i] / self.graph.sum_degrees)
            # SSEi = - ((self.graph.node_degrees[i] + self.graph_con.node_degrees[i])/self.graph.sum_degrees) * np.log2(self.graph.node_degrees[i]/self.graph.sum_degrees)
            ci = ({i}, self.graph.node_degrees[i], self.graph.node_degrees[i], self.graph_con.node_degrees[i], SSEi)  # nodes, volume, cut, cut', SSE
            self.communities[i] = ci
            self.SSE += SSEi
        for i in self.graph.adj.keys():
            for edge in self.graph.adj[i]:
                self.pair_cuts[frozenset([edge.i, edge.j])] = edge.weight
                self.connections[i].add(edge.j)
                self.connections[edge.j].add(i)
        for i in self.graph_con.adj.keys():
            for edge in self.graph_con.adj[i]:
                self.pair_cuts_con[frozenset([edge.i, edge.j])] = edge.weight
                self.connections_con[edge.i].add(edge.j)
                self.connections_con[edge.j].add(edge.i)

    def merge_deltaH_SSE_old(self, commID1, commID2):
        vertices1, v1, g1, g_con1, SSE1 = self.communities.get(commID1)
        vertices2, v2, g2, g_con2, SSE2 = self.communities.get(commID2)
        vx = v1 + v2
        gx = g1 + g2 - 2*self.pair_cuts.get(frozenset([commID1, commID2]))
        g_conx = g_con1 + g_con2
        if frozenset([commID1,commID2]) in self.pair_cuts_con:
            g_conx -= 2*self.pair_cuts_con.get(frozenset([commID1, commID2]))
        SSEx = cal_module_SSE(gx+g_conx, self.graph.sum_degrees, vx, self.graph.sum_degrees)
        for vertex in vertices1:
            SSEx += cal_module_SSE(self.graph.node_degrees[vertex], self.graph.sum_degrees,
                                   self.graph.node_degrees[vertex], vx)
        for vertex in vertices2:
            SSEx += cal_module_SSE(self.graph.node_degrees[vertex], self.graph.sum_degrees,
                                   self.graph.node_degrees[vertex], vx)
        return SSE1+SSE2-SSEx

    def merge_deltaH_SSE(self, commID1, commID2):
        vertices1, v1, g1, g_con1, SSE1 = self.communities.get(commID1)
        vertices2, v2, g2, g_con2, SSE2 = self.communities.get(commID2)
        vx = v1 + v2
        gx = g1 + g2 - 2 * self.pair_cuts.get(frozenset([commID1, commID2]))
        g_conx = g_con1 + g_con2
        if frozenset([commID1, commID2]) in self.pair_cuts_con:
            g_conx -= 2 * self.pair_cuts_con.get(frozenset([commID1, commID2]))
        deltaH = (v1-g1-g_con1)*np.log2(v1) + (v2-g2-g_con2)*np.log2(v2) - (vx-gx-g_conx)*np.log2(vx)
        deltaH += (g1+g_con1+g2+g_con2-gx-g_conx)*np.log2(self.graph.sum_degrees)
        deltaH /= self.graph.sum_degrees
        return deltaH

    def refinement_SSE(self):
        y, _ = self.communities2label()
        adj = self.A
        adj -= np.diag(np.diag(adj))
        W_con = self.A_con
        W_con -= np.diag(np.diag(W_con)).copy()
        tol = 1e-20
        max_iter = 300
        if y is None:
            n, k = adj.shape[0], 3
            y = np.random.randint(k, size=n)
        else:
            n, k = adj.shape[0], np.amax(y) + 1

        W = np.array(adj.copy(), dtype=np.float64)
        D = np.diag(np.sum(W, axis=-1, keepdims=False))
        D_con = np.diag(np.sum(W_con, axis=-1, keepdims=False))
        S = np.eye(k)[y.reshape(-1)].astype(np.float64)     # one hot of y
        volW = np.sum(W, dtype=np.float64)      # sum of degrees of A
        links_mtx = np.matmul(np.matmul(S.T, W), S)
        degree_mtx = np.matmul(np.matmul(S.T, D), S)
        links_con_mtx = np.matmul(np.matmul(S.T, W_con), S)
        degree_con_mtx = np.matmul(np.matmul(S.T, D_con), S)
        links = np.diagonal(links_mtx).copy()     # asso of each community, i.e., volume - cut
        degree = np.diagonal(np.clip(degree_mtx, a_min=EPS, a_max=None)).copy()
        links_con = np.diagonal(links_con_mtx).copy()
        degree_con = np.diagonal(np.clip(degree_con_mtx, a_min=EPS, a_max=None)).copy()
        # cuts_con = degree_con - links_con
        sses = ((-links+degree_con-links_con) / volW) * np.log2(np.clip(degree, a_min=1e-10, a_max=None) / volW)
        z = y.copy()
        sse = np.sum(sses)
        for iter_num in range(max_iter):
            for i in range(n):
                zi = z[i]
                links[zi] -= np.matmul(W[i, :], S[:, zi]) + np.matmul(S[:, zi].T, W[:, i])
                degree[zi] -= D[i, i]
                links_con[zi] -= np.matmul(W_con[i,:], S[:,zi]) + np.matmul(S[:,zi].T, W_con[:,i])
                degree_con[zi] -= D_con[i,i]
                sses[zi] = ((-links[zi]+degree_con[zi]-links_con[zi]) / volW) * np.log2(np.clip(degree[zi], a_min=1e-10, a_max=None) / volW)
                S[i, zi] = 0
                z[i] = -1

                links_new = links.copy()
                degree_new = degree.copy()
                links_new += np.matmul(W[i, :], S) + np.matmul(W[:, i].T, S)
                degree_new += D[i, i]
                links_con_new = links_con.copy()
                degree_con_new = degree_con.copy()
                links_con_new += np.matmul(W_con[i, :], S) + np.matmul(W_con[:, i].T, S)
                degree_con_new += D_con[i,i]
                sses_new = ((-links_new+degree_con_new-links_con_new) / volW) * np.log2(np.clip(degree_new, a_min=1e-10, a_max=None) / volW)
                delta_sses = sses_new - sses

                opt_i = np.argmax(delta_sses)

                zi = opt_i
                z[i] = zi
                S[i, zi] = 1
                links[zi] = float(links_new[zi])
                degree[zi] = float(degree_new[zi])
                links_con[zi] = float(links_con_new[zi])
                degree_con[zi] = float(degree_con_new[zi])
                sses[zi] = float(sses_new[zi])
            if np.sum(sses) - sse < tol:
                break
            sse = np.sum(sses)
        # z = self.remove_empty_cluster(z)
        # assert np.max(z)+1 == np.unique(z).shape[0]
        # k = np.max(z) + 1
        return z

    def remove_empty_cluster(self, z):
        z_new = np.zeros_like(z)
        label_old2new = dict()
        for i, label in enumerate(np.unique(z)):
            label_old2new[label] = i
        for i in range(z.shape[0]):
            z_new[i] = label_old2new[z[i]]
        return z_new

    def merge(self):
        merge_queue = []
        merge_map = dict()
        for pair in self.pair_cuts.keys():
            commID1, commID2 = pair
            if commID1 not in self.communities.keys() or commID2 not in self.communities.keys():
                continue
            deltaH = self.merge_deltaH_SSE(commID1, commID2)
            pair_mustlink = 0
            if self.mustlink_first:
                if frozenset([commID1,commID2]) in self.pair_cuts_con:
                    if self.pair_cuts_con[frozenset([commID1,commID2])] > 0:
                        pair_mustlink = 1
                    elif self.pair_cuts_con[frozenset([commID1,commID2])] < 0:
                        pair_mustlink = -1
                    else:
                        pair_mustlink = 0
                else:
                    pair_mustlink = 0
            merge_entry = [-pair_mustlink, -deltaH, pair]
            heapq.heappush(merge_queue, merge_entry)
            merge_map[pair] = merge_entry

        while len(merge_queue) > 0:
            pair_mustlink, deltaH, pair = heapq.heappop(merge_queue)
            pair_mustlink = - pair_mustlink
            deltaH = -deltaH
            if pair == frozenset([]):
                continue
            if deltaH<0:
                continue
            commID1, commID2 = pair
            if (commID1 not in self.communities) or (commID2 not in self.communities):
                continue
            self.SSE -= deltaH
            comm1 = self.communities.get(commID1)
            comm2 = self.communities.get(commID2)

            g_conx = comm1[3]+comm2[3]
            if frozenset([commID1,commID2]) in self.pair_cuts_con:
                g_conx -= 2*self.pair_cuts_con[frozenset([commID1,commID2])]
            new_comm = (comm1[0].union(comm2[0]), comm1[1]+comm2[1], comm1[2]+comm2[2]-2*self.pair_cuts[frozenset([commID1,commID2])],
                        g_conx, comm1[4]+comm2[4]-deltaH)
            self.communities[commID1] = new_comm
            self.communities.pop(commID2)

            if commID2 in self.connections_con[commID1]:
                self.connections_con[commID1].remove(commID2)
                self.connections_con[commID2].remove(commID1)
            for k in self.connections_con[commID1]:
                if k in self.connections_con[commID2]:
                    self.pair_cuts_con[frozenset([commID1,k])] = self.pair_cuts_con.get(frozenset([commID1,k])) + self.pair_cuts_con.get(frozenset([commID2,k]))
                    self.connections_con[commID2].remove(k)
                    self.connections_con[k].remove(commID2)
                    self.pair_cuts_con.pop(frozenset([commID2,k]))
            for k in self.connections_con[commID2]:
                self.pair_cuts_con[frozenset([commID1,k])] = self.pair_cuts_con[frozenset([commID2,k])]
                self.pair_cuts_con.pop(frozenset([commID2,k]))
                self.connections_con.get(k).remove(commID2)
                self.connections_con.get(k).add(commID1)
                self.connections_con.get(commID1).add(k)
            self.connections_con.get(commID2).clear()

            self.connections[commID1].remove(commID2)
            self.connections[commID2].remove(commID1)
            for k in self.connections[commID1]:
                if k in self.connections[commID2]:
                    pair_cut_1k = self.pair_cuts.get(frozenset([commID1,k])) + self.pair_cuts.get(frozenset([commID2,k]))
                    self.pair_cuts[frozenset([commID1,k])] = pair_cut_1k
                    self.connections[commID2].remove(k)
                    self.connections[k].remove(commID2)
                    self.pair_cuts.pop(frozenset([commID2, k]))
                    merge_entry = merge_map.pop(frozenset([commID2, k]))
                    merge_entry[-1] = frozenset([])
                else:
                    pair_cut_1k = self.pair_cuts[frozenset([commID1,k])]
                deltaH1k = self.merge_deltaH_SSE(commID1,k)
                merge_entry = merge_map.pop(frozenset([commID1,k]))
                merge_entry[-1] = frozenset([])
                pair_mustlink = 0
                if self.mustlink_first:
                    if frozenset([commID1, commID2]) in self.pair_cuts_con:
                        if self.pair_cuts_con[frozenset([commID1, commID2])] > 0:
                            pair_mustlink = 1
                        elif self.pair_cuts_con[frozenset([commID1, commID2])] < 0:
                            pair_mustlink = -1
                        else:
                            pair_mustlink = 0
                    else:
                        pair_mustlink = 0
                merge_entry = [-pair_mustlink, -deltaH1k, frozenset([commID1,k])]
                heapq.heappush(merge_queue,merge_entry)
                merge_map[frozenset([commID1,k])] = merge_entry
            for k in self.connections[commID2]:
                self.pair_cuts[frozenset([commID1,k])] = self.pair_cuts[frozenset([commID2,k])]
                self.pair_cuts.pop(frozenset([commID2,k]))
                deltaH1k = self.merge_deltaH_SSE(commID1,k)
                merge_entry = merge_map.pop(frozenset([commID2,k]))
                merge_entry[-1] = frozenset([])
                pair_mustlink = 0
                if self.mustlink_first:
                    if frozenset([commID1, commID2]) in self.pair_cuts_con:
                        if self.pair_cuts_con[frozenset([commID1, commID2])] > 0:
                            pair_mustlink = 1
                        elif self.pair_cuts_con[frozenset([commID1, commID2])] < 0:
                            pair_mustlink = -1
                        else:
                            pair_mustlink = 0
                    else:
                        pair_mustlink = 0
                merge_entry = [-pair_mustlink, -deltaH1k, frozenset([commID1,k])]
                heapq.heappush(merge_queue, merge_entry)
                merge_map[frozenset([commID1,k])] = merge_entry
                self.connections.get(k).remove(commID2)
                self.connections.get(k).add(commID1)
                self.connections.get(commID1).add(k)
            self.connections.get(commID2).clear()
            # self.connections.pop(commID2)

    def build_tree(self):
        self.init_encoding_tree()
        self.merge()
        # y, _ = self.communities2label()
        y = self.refinement_SSE()
        return y

    def communities2label(self):
        y_pred = np.zeros(self.graph.num_nodes, dtype=int)
        label2commID = dict()
        for i, ci in enumerate(sorted(self.communities.keys())):
            y_pred[np.array(list(self.communities[ci][0])).astype(int)] = i
            label2commID[i] = ci
        return y_pred, label2commID




if __name__=='__main__':
    graph = read_graph("E:/constrained_clustering/constrainedSE/lymph6graph/Lymph6Graph")
    graph_con = Graph(graph.num_nodes)
    flatSSE = FlatSSE(graph, graph_con)
    flatSSE.build_tree()
    print(flatSSE.communities)
    print(flatSSE.SSE)