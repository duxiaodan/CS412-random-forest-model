#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:56:17 2018

@author: xiaodan
"""

import sys

class Node:
    def __init__(self, best_attri_index, best_attri_index_value, best_Gini, best_partitions):
        self.attri_index = best_attri_index
        self.attri_index_value = best_attri_index_value
        self.Gini = best_Gini
        self.partition_0 = best_partitions[0]
        self.partition_1 = best_partitions[1]
        self.child_node_0 = None
        self.child_node_1 = None
    def clear_partitions(self):
        self.partition_0=[]
        self.partition_1=[]
    def add_child(self,child_node,n):
        if n==0:
            self.child_node_0 = child_node
        elif n==1:
            self.child_node_1 = child_node

def gini(partition):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    num_data_point=len(partition)
    if num_data_point==0:
        return 1
    sum_sq=0
    for i in range(1,num_classes+1):
        count_curr_class=len(list(filter(lambda x: x[0] == i, partition)))
        sum_sq+=(count_curr_class/num_data_point)**2
    return 1-sum_sq

def Gini(partitions):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    total_num=sum(map(len, partitions.values()))
    sum_gini=0
    for key, value in partitions.items():
        sum_gini+=(len(value)/total_num)*gini(value)
    return sum_gini

def get_partitions(data,attri_index):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    partitions={}
    for i in range(1,class_values[attri_index]+1):
        partitions[i]={0:[],1:[]}
        for data_point in data:
            if data_point[attri_index]==i:
                partitions[i][1].append(data_point)
            else:
                partitions[i][0].append(data_point)
    # partitions structure:{value_1:{0:list,1:list},value_2:{0:list,1:list},...,value_n:{0:list,1:list}}
    return partitions

def get_best_partitions(data):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    best_attri_index=0
    best_attri_index_value=0
    best_Gini=sys.float_info.max
    best_partitions={}
    for attri_index in class_values.keys():
        partitions=get_partitions(data,attri_index)
        for attri_value, partition in partitions.items():
            Gini_value=Gini(partition)
            if Gini_value<best_Gini:
                best_attri_index=attri_index
                best_attri_index_value=attri_value
                best_Gini=Gini_value
                best_partitions=partition
    #best_partitions structure: {0:list,1:list}
    return Node(best_attri_index, best_attri_index_value, best_Gini, best_partitions)

def leaf_terminate(data):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    class_list=[data_point[0] for data_point in data]
    return max(set(class_list), key=class_list.count)

def splitting(node,depth):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    new_d1=0
    new_d2=0
    partition_0=node.partition_0
    partition_1=node.partition_1
    node.clear_partitions
    if len(partition_0)==0 or len(partition_1)==0:
        leaf_decision = leaf_terminate(partition_0+partition_1)
        node.add_child(leaf_decision,0)
        node.add_child(leaf_decision,1)
        return depth+1
    if depth>=max_tree_depth-1:
        node.add_child(leaf_terminate(partition_0),0)
        node.add_child(leaf_terminate(partition_1),1)
        return depth+1
    if len(partition_0)>min_node_size:
        node.add_child(get_best_partitions(partition_0),0)
        new_d1=splitting(node.child_node_0,depth+1)
    else:
        node.add_child(leaf_terminate(partition_0),0)
    if len(partition_1)>min_node_size:
        node.add_child(get_best_partitions(partition_1),1)
        new_d2=splitting(node.child_node_1,depth+1)
    else:
        node.add_child(leaf_terminate(partition_1),1)
    return max(new_d1,new_d2)
        
def decision_tree_construction(data):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    Root=get_best_partitions(data)
    depth = splitting(Root,0)
    return Root,depth

def visualize_tree(node, depth=0):
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    if isinstance(node, Node):
        print('%s[Attri%d = %d]' % ((depth*' ', (node.attri_index), node.attri_index_value)))
        visualize_tree(node.child_node_0, depth+1)
        visualize_tree(node.child_node_1, depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

def predict_single_data(data_point,node):
    if data_point[node.attri_index]==node.attri_index_value:
        if isinstance(node.child_node_1,int):
            return node.child_node_1
        else:
            return predict_single_data(data_point,node.child_node_1)
    else:
        if isinstance(node.child_node_0,int):
            return node.child_node_0
        else:
            return predict_single_data(data_point,node.child_node_0)
def predict(data,root,confusion_matrix):
    stats=[]
    for data_point in data:
        pred=predict_single_data(data_point,root)
        stats.append([data_point[0],pred,data_point[0]==pred])
        confusion_matrix[data_point[0]-1][pred-1]+=1
    for i,row_i in enumerate(confusion_matrix):
        for j,item_j in enumerate(row_i):
            confusion_matrix[i][j]=str(confusion_matrix[i][j])
    return stats,confusion_matrix

def print_confusion_matrix(matrix):
    for line in matrix:
        text=' '.join(line)
        sys.stdout.write(text+'\n')
        
def main():
    #Data parsing
    global class_values
    global num_attri
    global num_classes
    global max_tree_depth
    global min_node_size
    script = sys.argv[0]
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    if 'balance.scale' in training_file:
        max_tree_depth=15
        min_node_size=6
    if 'nursery' in training_file:
        max_tree_depth=15
        min_node_size=2
    if 'led' in training_file:
        max_tree_depth=8
        min_node_size=8
    if 'synthetic.social' in training_file:
        max_tree_depth=20
        min_node_size=5
    f1=open(training_file,'r')
    training_data_r=f1.readlines();
    f1.close()
    num_classes=0 # number of classes
    
    class_values={} # how many values are there for a given attribute
    training_data=[]
    for line in training_data_r:
        line=line.strip().split(' ')
        curr_line=[]
        for item in line:
            curr_line.append(int(item.split(':')[-1]))
        training_data.append(curr_line)
        if curr_line[0] > num_classes:
            num_classes=curr_line[0]
        for i in range(1,len(curr_line)):
            if i in class_values:
                class_values[i]=max(curr_line[i],class_values[i])
            else:
                class_values[i]=curr_line[i]
    f2=open(testing_file,'r')
    testing_data_r=f2.readlines();
    f2.close()
    testing_data=[]

    for line in testing_data_r:
        line=line.strip().split(' ')
        curr_line=[]
        for item in line:
            curr_line.append(int(item.split(':')[-1]))
        testing_data.append(curr_line)
    num_attri=len(training_data[0])-1
    #Building decision tree with gini-index
    tree,depth=decision_tree_construction(training_data)
#    print('depth',depth)
    confusion_matrix_empty=[]
    for i_class in range(1,num_classes+1):
        row = []
        for j_class in range(1,num_classes+1):
            row.append(0)
        confusion_matrix_empty.append(row)
    stats,confusion_matrix=predict(testing_data,tree,confusion_matrix_empty)
    percentage=sum(map(lambda x: x[2], stats))/len(stats)
#    print('total',len(stats))
#    print(percentage)
#    print(confusion_matrix)
#    print(testing_data[20:35])
#    print(class_values,num_attri,num_classes)
    print_confusion_matrix(confusion_matrix)
if __name__ == '__main__':
   main()