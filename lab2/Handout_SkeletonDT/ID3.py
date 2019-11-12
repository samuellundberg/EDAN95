from collections import Counter
from graphviz import Digraph
from math import log2


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
    def find_split_attr(self, data, attributes, classes, target, cls_freq):
        max_IG = 100000
        best_attr = ''
        a_count = 0
        a_count_best = 0


        for a in attributes:
            # measure entropy for a
            feature_dict = {}
            for d in data:
                if d[a_count] in feature_dict.keys():
                    feature_dict[d[a_count]]['count'] += 1  # value of attribute for data
                    # for each attribute we want to know how many of these
                    # that belonged to a certain class
                    feature_dict[d[a_count]][target] += 1
                else:
                    feature_dict[d[a_count]] = {'count': 1}
                    feature_dict[d[a_count]]['count'] = 1
                    for c in classes:
                        feature_dict[d[a_count]][c] = 0
                    feature_dict[d[a_count]][target] += 1
            # now we have all information about current attribute...
            # TODO: Calculate Information Gain
            IG = 0
            for key in feature_dict.keys():
                S = len(data)
                Sv = feature_dict[key]['count']
                Isv = 0
                for c in classes:
                    Isv -= feature_dict[key][c] / Sv * log2(feature_dict[key][c] / Sv)
                IG += Sv / S * Isv  # we want to find a that minimize IG
            if IG < max_IG:
                max_IG = IG
                best_attr = a
                a_count_best = a_count
            a_count += 1
        # Change this to make some more sense
        return best_attr, a_count_best


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
            self.add_node_to_graph(root)
            return root

        # if attribute empty
        if len(attributes) == 0:
            cls_freq = {}
            for t in target:
                if t in cls_freq.keys:
                    cls_freq[t] += 1
                else:
                    cls_freq[t] = 1
            max_freq = 0
            max_class = ''
            for c in cls_freq:
                if c.value() > max_freq:
                    max_freq = c.value()
                    max_class = c.key()

            root['label'] = max_class
            return root
        else:

# BEGIN
            cls_freq = {}
            # count
            for t in target:
                if t in cls_freq.keys:
                    cls_freq[t] += 1
                else:
                    cls_freq[t] = 1
            #          Set A as the target_attribute of Root
            root['attribute'], a_count = self.find_split_attr(data, attributes, classes, target, cls_freq)
            new_nodes = []
            for v in attributes[root['attribute']]:
                #  add a new tree branch below Root
                new_data = []
                new_target = [] # make tuple when done
                for i, d in enumerate(data):
                    if d[a_count] == v:
                        new_d = tuple(list(d[:a_count]) + list(d[a_count+1:]))
                        new_data.append(new_d)
                        new_target.append(target[i])
                new_target = tuple(new_target)

                # could be outside loop
                new_attributes = {}
                for key in attributes.keys():
                    if key != root['attribute']:
                        new_attributes[key] = attributes[key]
                # TODO
                new_classes = []
                new_classes = tuple(new_classes)

                node = self.fit()
                #self, data, target, attributes, classes):


        # fill in something more sensible here...
        # root should become the output of the recursive tree creation

        root = self.new_ID3_node()
        self.add_node_to_graph(root)

        return root



    def predict(self, data, tree) :
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted