import ToyData as td
import ID3

import numpy as np
# from sklearn import tree, metrics, datasets


def main():
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    print(classes)
    print(target)
    print(attributes)
    print(data)
    id3 = ID3.ID3DecisionTreeClassifier()

    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, myTree, attributes)
    print('predicted: ', predicted)
    print('data2: ', data2)
    print('target2: ', target2)


if __name__ == "__main__": main()