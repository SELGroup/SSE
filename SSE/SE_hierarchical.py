import copy
import math
import heapq
import numba as nb
import numpy as np
import networkx as nx
from ete3 import Tree
from queue import Queue
import copy

def get_id():
    i = 0
    while True:
        yield i
        i += 1

def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i,j] != 0:
                n_v += adj_matrix[i,j]
                VOL += adj_matrix[i,j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes,VOL,node_vol,adj_table

@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i],p2[j]]
            if c != 0:
                c12 += c
    return c12

def LayerFirst(node_dict,start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)

def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
                                 g=g, vol=v,child_h= child_h,child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node

def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break

def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth+=1
    deepth += node_dict[nid].child_h
    return deepth



class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children:set = None,parent = None,child_h = 0, child_cut = 0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree():
    def __init__(self,adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.SE = 0
        self.build_leaves()

    def CombineDelta(self, node1, node2, cut_v, g_vol):
        v1 = node1.vol
        v2 = node2.vol
        g1 = node1.g
        g2 = node2.g
        v12 = v1 + v2
        return -(2 * cut_v / g_vol) * np.log2(g_vol / v12)

    def CompressDelta(self, node1, p_node, g_vol):
        assert node1.children is not None
        children = node1.children
        cut_sum = 0
        for child in children:
            cut_sum += self.tree_node[child].g
        return -((cut_sum - node1.g) / g_vol) * np.log2(node1.vol / p_node.vol)

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)
            # self.root_node.children.add(ID)
            self.SE -= (v/self.VOL) * np.log2(v/self.VOL)

    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                ent += - (node_g / self.VOL) * np.log2(node_vol / node_p_vol)
        return ent

    def __build_k_tree(self, g_vol, nodes_dict:dict, k=None,):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
                    diff = self.CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            # assert diff<0
            # if diff > 0:
            #     continue
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.SE += diff
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            #compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,[self.CompressDelta(nodes_dict[id1],nodes_dict[new_id], g_vol),id1,new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,[self.CompressDelta(nodes_dict[id2],nodes_dict[new_id], g_vol),id2,new_id])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix,np.array(n1.partition), np.array(n2.partition))

                    new_diff = self.CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,partition=list(range(self.g_num_nodes)),children=unmerged_nodes,
                                         vol=g_vol,g = 0,child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [self.CompressDelta(nodes_dict[i], nodes_dict[new_id], g_vol), i, new_id])
            root = new_id
        tree_node_copy = copy.deepcopy(self.tree_node)

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                self.SE += diff
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = self.CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]], g_vol)
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = self.CompressDelta(nodes_dict[e[1]], nodes_dict[p_id], g_vol)
                heapq.heapify(cmp_heap)
        return root, tree_node_copy

    def coding_tree_2ete3(self, root, tree_node):
        t = Tree()
        root_ete = t.add_child(name=str(root))
        q = Queue()
        q.put((root, root_ete))
        while q.qsize()>0:
            node, node_ete = q.get()
            if tree_node[node].children is None:
                continue
            for node_i in tree_node[node].children:
                node_i_ete = node_ete.add_child(name=str(node_i))
                q.put((node_i, node_i_ete))
        # print(t)
        return t


    def build_coding_tree(self, k=2, mode='v1'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id, hierarchical_tree_node = self.__build_k_tree(self.VOL, self.tree_node, k=k)
            return self.root_id, hierarchical_tree_node

