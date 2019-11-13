import numpy as np
from collections import Counter
from graphviz import Digraph

class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes, cls_freq):
        max_IG = 100000
        best_attr = ''
        a_count = 0
        a_count_best = 0
        if len(attributes) == 1:
            print(list(attributes.keys())[0])
            return list(attributes.keys())[0], 0

        for a in attributes:
            # measure entropy for a
            feature_dict = {}
            for i, d in enumerate(data):
                if d[a_count] in feature_dict.keys():
                    feature_dict[d[a_count]]['count'] += 1  # value of attribute for data
                    # for each attribute we want to know how many of these
                    # that belonged to a certain class
                    feature_dict[d[a_count]][target[i]] += 1
                else:
                    feature_dict[d[a_count]] = {'count': 1}
                    feature_dict[d[a_count]]['count'] = 1
                    for c in classes:
                        feature_dict[d[a_count]][c] = 0
                    feature_dict[d[a_count]][target[i]] += 1
            # now we have all information about current attribute...
            # TODO: Calculate Information Gain
            IG = 0
            for key in feature_dict.keys():
                S = len(data)
                Sv = feature_dict[key]['count']
                Isv = 0
                for c in classes:
                    if feature_dict[key][c] / Sv != 0:
                        Isv -= feature_dict[key][c] / Sv * np.log2(feature_dict[key][c] / Sv)

                IG += Sv / S * Isv  # we want to find a that minimize IG
            if IG < max_IG:
                max_IG = IG
                best_attr = a
                a_count_best = a_count
            a_count += 1
        # Change this to make some more sense
        return best_attr, a_count_best

    def find_most_common_class(self, target):
        freq_classes = {}
        for t in target:
            if t in freq_classes.keys():
                freq_classes[t] += 1
            else:
                freq_classes[t] = 0

        max_class = ''
        freq_class = 0
        for key in freq_classes.keys():
             if freq_classes[key] > freq_class:
                 freq_class = freq_classes[key]
                 max_class = key
        print('HERE')
        print(max_class)
        return max_class


    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        root = self.new_ID3_node()

        # If all belongs to one class only return one node
        one_class = True
        for t in target:
            if t != target[0]:
                one_class = False
                break

        if one_class:
            root['label'] = target[0]
            return root

        # if attribute empty
        if len(attributes) == 0:
            cls_freq = {}
            for t in target:
                if t in cls_freq.keys():
                    cls_freq[t] += 1
                else:
                    cls_freq[t] = 1
            max_freq = 0
            max_class = ''
            for c in cls_freq:
                print(cls_freq[c])
                if cls_freq[c] > max_freq:
                    max_freq = cls_freq[c]
                    max_class = c

            root['label'] = max_class
            self.add_node_to_graph(root)

            return root
        else:

# BEGIN
            cls_freq = {}
            # count
            for t in target:
                if t in cls_freq.keys():
                    cls_freq[t] += 1
                else:
                    cls_freq[t] = 1
            #          Set A as the target_attribute of Root
            root['attribute'], a_count = self.find_split_attr(data, target, attributes, classes, cls_freq)
            new_nodes = []
            print('here')
            print(root['attribute'])
            print(root)
            print(attributes)
            # go through one attribute: {'color': ['y', 'g', 'b']}
            for v in attributes[root['attribute']]:
                #  add a new tree branch below Root
                new_data = []
                new_target = []     # make tuple when done
                new_classes = []    # make tuple when done
                # create subdata for current value of current attribute, eg all w red color
                for i, d in enumerate(data):
                    if d[a_count] == v:
                        new_d = tuple(list(d[:a_count]) + list(d[a_count+1:]))
                        new_data.append(new_d)
                        new_target.append(target[i])
                        if target[i] not in new_classes:
                            new_classes.append(target[i])

                new_target = tuple(new_target)
                new_classes = tuple(new_classes)

                # could be outside loop
                new_attributes = {}
                for key in attributes.keys():
                    if key != root['attribute']:
                        new_attributes[key] = attributes[key]

                if len(new_data) == 0: #If Samples(v) is empty, then
                    # add new branch with leaf as label  = most common class value in Samples.
                    leaf = self.new_ID3_node()
                    leaf['label'] = self.find_most_common_class(target) #think this is correct
                    root['nodes'] = leaf
                    self.add_node_to_graph(leaf, root['id'])

                else:
                    print("vi gör recursion!!!")
                    node = self.fit(new_data, new_target, new_attributes, new_classes)
                    new_nodes.append(node)
                    root['nodes'] = new_nodes
                    self.add_node_to_graph(node,root['id'])
                    self.add_node_to_graph(root)



        # fill in something more sensible here...
        # root should become the output of the recursive tree creation

        self.add_node_to_graph(root)

        return root



    def predict(self, data, tree) :
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted