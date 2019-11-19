import graphviz
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import tree
import ID3

from sklearn import metrics

if __name__ == '__main__':
# 1. Use the code skeleton / snippets provided or the notebook used in the tutorial
# 	from the first course week to load the digits dataset from the datasets provided
# 	in SciKitLearn. Inspect the data. What is in there?

#   The set contains images of digits and labels

    # Load the digits dataset
    digits = datasets.load_digits()

    # Display the first digit
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    #plt.show()
    print(digits.data[0])
    # Display one image for each digit
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(2, 5, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)
    #plt.show()

#	2. Split your data set into 70% training data (features and labels), and 30% test 	data.

    split = int(len(digits.data)*0.7)

    training_feature = digits.data[:split]
    test_feature = digits.data[split+1:]
    training_labels = digits.target[:split]
    test_labels = digits.target[split + 1:]

# 3. Set up a DecisionTreeClassifier as it comes in SciKitLearn.
# TODO: Follow the tutorial (or the respective documentation) and produce a
#  plot of the tree with graphviz.
#  What can you learn from this about how the used algorithm handles the data?
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(training_feature, training_labels)


    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("digitTree_min_leaf_4")
    # The algorithm checks the value of one position if its larger than x

    y_predict = classifier.predict(test_feature)

    print(metrics.confusion_matrix(test_labels, y_predict))
    print(metrics.classification_report(test_labels, y_predict))
    # 79% accuracy


    id3 = ID3.ID3DecisionTreeClassifier()


    print(training_feature[0])
    print(type(training_feature[0][0]))

    attributes = {}
    att_l = []
    for i in range(16):
        att_l.append(float(i))

    for i in range(len(training_feature[0])):
        attributes[i] = att_l

    classes = []
    for i in range(10):
        classes.append(i)

    myTree = id3.fit(training_feature, training_labels, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(test_feature, myTree, attributes)
    print(predicted)
    print(test_labels)

    print(metrics.confusion_matrix(test_labels, predicted))
    print(metrics.classification_report(test_labels, predicted))

    new_data = []
    new_test_data = []
    for im in training_feature:
        new_im = []
        for pix in im:
            if pix < 5:
                new_im.append('black')
            elif pix < 10:
                new_im.append('grey')
            else:
                new_im.append('white')
        new_data.append(new_im)

    for im in test_feature:
        new_im = []
        for pix in im:
            if pix < 5:
                new_im.append('black')
            elif pix < 10:
                new_im.append('grey')
            else:
                new_im.append('white')
        new_test_data.append(new_im)
    print(new_data)
    new_attributes = {}
    for i in range(len(training_feature[0])):
        new_attributes[i] = ['black', 'grey', 'white']
    id3_2 = ID3.ID3DecisionTreeClassifier()

    myTree = id3_2.fit(new_data, training_labels, new_attributes, classes)
    plot2 = id3_2.make_dot_data()
    plot2.render("testTree2")
    predicted = id3_2.predict(new_test_data, myTree, new_attributes)

    print(metrics.confusion_matrix(test_labels, predicted))
    print(metrics.classification_report(test_labels, predicted))