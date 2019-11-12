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
    def find_split_attr(self):

        # Change this to make some more sense
        return None


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
            cls_freq = {}
            for t in target:
                if t in cls_freq.keys:
                    cls_freq[t] += 1
                else:
                    cls_freq[t] = 1
            while True:
                max_entr = 0
                max_attr = ''
                a_count = 0
                for a in attributes:
                    # measure entropy for a
                    feature_dict = {}
                    for d in data:
                        if d[a_count] in feature_dict.keys():
                            feature_dict[d[a_count]]['count'] += 1 #value of attribute for data
                            # for each class count how many
                            # if in keys.. 
                            # feature_dict[d[a_count]][target]
                        else:
                            feature_dict[d[a_count]]['count'] = 1

                        entropy = 0
                        if entropy > max_entr:
                            max_attr = a
                            max_entr = entropy
                    a_count += 1

        # fill in something more sensible here...
        # root should become the output of the recursive tree creation

        root = self.new_ID3_node()
        self.add_node_to_graph(root)

        return root



    def predict(self, data, tree) :
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted