import pandas as pd
import numpy as np
import math
from timeit import default_timer as timer


class check:
    def __init__(self, condition, value):
        self.condition = condition
        self.value = value


class node:
    def __init__(self, name, column, question, left_child, right_child, gain):
        self.name = "Node" + str(name)
        self.left_child = left_child
        self.right_child = right_child
        self.check = question
        self.column = column
        self.ig_gain = gain


class output:
    def __init__(self, data):
        self.output = data.groupby(label)[label].count()


def partition(data, col, value):
    left = data[data[col] >= value]
    right = data[data[col] < value]
    return left, right


def gini_index(data):
    label_distribuition = data.groupby(label)[label].count()
    return 1 - (((label_distribuition / data.shape[0]) ** 2).sum())


def entropy_value(data):
    label_distribuition = data.groupby(label)[label].count()
    probs = label_distribuition / data.shape[0]
    entropy = 0
    for p in probs:
        entropy += (p * math.log(1 / p, 2))
    return entropy


def best_split_gini(data):
    # q_data = data.quantile([0.25, 0.5, 0.75])
    gini = {}
    Gain_value = {}
    Gain_question = {}
    for col in columns:
        gini[col] = gini_index(data)
        Gain_value[col] = 0
        Gain_question[col] = 0
        # for question in q_data[col]:
        for question in data[col]:
            left, right = partition(data, col, question)
            left_gini = gini_index(left)
            right_gini = gini_index(right)
            ave_gini_after_split = (left.shape[0] / data.shape[0] * left_gini) + (
                    right.shape[0] / data.shape[0] * right_gini)
            if Gain_value[col] < gini[col] - ave_gini_after_split:
                Gain_value[col] = gini[col] - ave_gini_after_split
                Gain_question[col] = question
    col = None
    for key, value in Gain_value.items():
        if value == max(Gain_value.values()):
            col = key
            break

    return col, Gain_question[col], Gain_value[col]


def best_split_entropy(data):
    # q_data = data.quantile([0.25, 0.5, 0.75])
    entropy = {}
    IG_value = {}
    IG_question = {}
    for col in columns:
        entropy[col] = entropy_value(data)
        IG_value[col] = 0
        IG_question[col] = 0
        # for question in q_data[col]:
        for question in data[col]:
            left, right = partition(data, col, question)
            left_entropy = entropy_value(left)
            right_entropy = entropy_value(right)
            if IG_value[col] < (entropy[col] - (
                    (left.shape[0] / data.shape[0] * left_entropy) + (right.shape[0] / data.shape[0] * right_entropy))):
                IG_value[col] = (entropy[col] - ((left.shape[0] / data.shape[0] * left_entropy) + (
                        right.shape[0] / data.shape[0] * right_entropy)))
                IG_question[col] = question
    col = None
    for key, value in IG_value.items():
        if value == max(IG_value.values()):
            col = key
            break
    return col, IG_question[col], IG_value[col]


def create_gini_tree(data):
    global node_name
    node_name += 1
    name = node_name
    gini_col, gini_question, gini_value = best_split_gini(data)
    if gini_value == 0:
        return output(data)
    left, right = partition(data, gini_col, gini_question)
    left_child = create_gini_tree(left, )
    right_child = create_gini_tree(right)

    return node(name, gini_col, gini_question, left_child, right_child, gini_value)


def create_entropy_tree(data):
    global node_name
    node_name += 1
    name = node_name
    entropy_col, entropy_question, ig_value = best_split_entropy(data)
    if ig_value == 0:
        return output(data)
    left, right = partition(data, entropy_col, entropy_question)
    left_child = create_entropy_tree(left)
    right_child = create_entropy_tree(right)

    return node(name, entropy_col, entropy_question, left_child, right_child, ig_value)


def print_tree(node, spacing=""):
    if isinstance(node, output):
        print(spacing + "Predict", node.output)
        return
    print(spacing + node.name + "(" + str(node.column) + ",Gain=" + str(format(node.ig_gain, '0.5f')) + ")")
    print(spacing + '--> True:')
    print_tree(node.left_child, spacing + "    ")
    print(spacing + '--> False:')
    print_tree(node.right_child, spacing + "    ")


def print_output(actual, output):
    global accuracy
    if len(output) == 1:
        for lbl in output.keys():
            v = int(lbl)
        if actual == v:
            accuracy += 1
            return str(v)

        return str(v) + "  Incorrect"
    total = sum(output) * 1.0
    probs = {}
    for lbl in output.keys():
        probs[lbl] = str(int(output[lbl] / total * 100)) + "%"
    return probs


def predict(row, node):
    if isinstance(node, output):
        return node.output
    if row[node.column] >= node.check:
        return predict(row, node.left_child)
    else:
        return predict(row, node.right_child)


if __name__ == "__main__":
    global columns, label, node_name, accuracy

    # orig_data = pd.read_excel(r"Indian_Liver_Patient_Dataset_ILPD.xlsx", sheet_name=None)
    #orig_data = pd.read_excel(r"pima_dataset.xlsx", sheet_name=None)
    #orig_data = pd.read_excel(r"C:\Users\Aditya\PycharmProjects\DM ASS1\Assignment2\ionosphere.csv.xlsx",
    # sheet_name=None)
    # orig_data = pd.read_excel(r"Immunotherapy.xlsx", sheet_name=None)

    # sheet = str(list(orig_data.keys())[0])
    # orig_data = orig_data[sheet]
    # orig_data = orig_data.dropna()
    # for col in orig_data.columns:
    #     if not (orig_data[col].dtype == np.int64 or orig_data[col].dtype == float):
    #         print("Dataset is not numeric")
    orig_data = pd.read_csv(r"C:\Users\Aditya\PycharmProjects\DM ASS1\Assignment2\ionosphere - Copy.csv", sep=",", header=None, engine='c')
    columns = np.asarray(orig_data.columns)
    label, columns = columns[-1], np.delete(columns, -1)

    """Kfold: Splitting Data into train:70% test:30%"""
    n = orig_data.shape[0]
    gini_avg_accuracy = 0
    IG_avg_accuracy = 0
    for i in range(1):
        print("\n-------------- ITERATION %d --------------" % (i + 1))
        train_data = orig_data.sample(n=int(n * 0.5))
        test_data = orig_data
        test_data = test_data.drop(train_data.index)

        start = timer()
        node_name = 0
        gini_tree = create_gini_tree(train_data)
        gini_nodes_created = node_name

        #print("Gini Tree:\n\n")
        #print_tree(gini_tree)

        accuracy = 0
        print("Gini Decision Tree Created: (time taken:", format(timer() - start, '0.2f'), "s , dataset length:", n,
              ")")
        for index, row in test_data.iterrows():
            #print("Actual: %s. Predicted: %s" % (int(row[-1]), print_output(int(row[-1]), predict(row, gini_tree))))
            print_output(int(row[-1]), predict(row, gini_tree))
        print("Gini nodes created:", gini_nodes_created)
        print("Correct predictions: ", str(accuracy) + " / " + str(test_data.shape[0]),
              format(accuracy / test_data.shape[0] * 100, '0.2f'), "%")
        gini_avg_accuracy = (accuracy + gini_avg_accuracy)

        start = timer()
        node_name = 0
        entropy_tree = create_entropy_tree(train_data)
        entropy_nodes_created = node_name

        #print("Entropy Tree:\n\n")
        #print_tree(entropy_tree)

        accuracy = 0
        print("IG Decision Tree Created: (time taken:", format(timer() - start, '0.2f'), "s , dataset size:", n, ")")
        for index, row in test_data.iterrows():
            # print("Actual: %s. Predicted: %s" % (int(row[-1]), print_output(int(row[-1]), predict(row, gini_tree))))
            print_output(int(row[-1]), predict(row, entropy_tree))
        print("Entropy nodes created:", entropy_nodes_created)
        print("Correct predictions: ", str(accuracy) + " / " + str(test_data.shape[0]),
              format(accuracy / test_data.shape[0] * 100, '0.2f'), "%")
        IG_avg_accuracy = (accuracy + IG_avg_accuracy)

    print("\nGini Average Accuracy:", format(gini_avg_accuracy/5 / (n/2) * 100, '0.2f'))
    print("IG Average Accuracy:", format(IG_avg_accuracy/5 / (n/2) * 100, '0.2f'))
