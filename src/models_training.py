import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
import json

features = [
    "bow_unigrams",
    "bow_bigrams",
    "bow_trigrams",
    "pos_unigrams",
    "pos_bigrams",
    "word_couples",
    "entities",
    "adverbs",
    "verbs",
    "nouns",
    "modal_auxs",
    "punctuation",
    "key_words",
    "text_length",
    "text_position",
    "token_count",
    "avg_word_length",
    "punct_marks_count",
    "parse_tree_depth",
    "sub_clauses_count",
    "clase",
]


class Results:
    """Class to store the results of the evaluation of the models. The results are the accuracy, the confusion matrix, the log loss, the precision, the recall and the F1 score."""

    def __init__(
        self, accuracy, confusion_matrix, precision, recall, f1_score, best_epoch=None
    ):
        """Initialize the results with the accuracy, the confusion matrix, the log loss, the precision, the recall and the F1 score."""
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.best_epoch = best_epoch

    def __str__(self):
        """Return the results as a string."""
        formatted_confusion_matrix_without_newlines = str(
            self.confusion_matrix
        ).replace("\n", " ")
        return f"Accuracy: {self.accuracy}, Confusion Matrix: {formatted_confusion_matrix_without_newlines}, Precision: {self.precision}, Recall: {self.recall}, F1 Score: {self.f1_score}"


class NeuralNetwork:
    """Class to store the neural network with the number of units for the hidden layers."""

    first_layer_units: int
    second_layer_units: int
    third_layer_units: int

    def __init__(
        self, first_layer_units: int, second_layer_units: int, third_layer_units: int
    ):
        """Initialize the neural network with the number of units for the hidden layers."""
        self.first_layer_units = first_layer_units
        self.second_layer_units = second_layer_units
        self.third_layer_units = third_layer_units

    def get_first_layer_units(self):
        """Return the number of units for the first hidden layer."""
        return self.first_layer_units

    def get_second_layer_units(self):
        """Return the number of units for the second hidden layer."""
        return self.second_layer_units

    def get_third_layer_units(self):
        """Return the number of units for the third hidden layer."""
        return self.third_layer_units


def load_features(file_path):
    """Load the features of the texts from a JSON file."""

    data_rows = []

    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            line = line.strip()
            if line:
                row = {}
                data = json.loads(line)
                for feature in features:
                    value = data[feature]
                    if feature in [
                        "bow_unigrams",
                        "bow_bigrams",
                        "bow_trigrams",
                        "pos_unigrams",
                        "pos_bigrams",
                        "word_couples",
                        "entities",
                        "adverbs",
                        "verbs",
                        "nouns",
                        "modal_auxs",
                        "punctuation",
                        "key_words",
                    ]:
                        value = len(data[feature])
                    row[feature] = value
                data_rows.append(row)
        file.close()
    return data_rows


def cross_validation_splits(data):
    """Split the data into num_splits splits for cross validation, splitting the data into training and test data. Return a list of tuples with the training and test data."""
    splits = []
    num_splits = 5
    split_train_test_size = len(data) // num_splits
    splits_train_test = []
    for i in range(num_splits):
        if i != num_splits - 1:
            sp = data.iloc[i * split_train_test_size : (i + 1) * split_train_test_size]
        else:
            sp = data.iloc[i * split_train_test_size :]
        splits_train_test.append(sp)
    for i in range(num_splits):
        test_data = splits_train_test[i]
        train_data = pd.concat(
            [splits_train_test[j] for j in range(num_splits) if j != i]
        )
        splits.append((train_data, test_data))
    return splits


def cross_validation_splits_with_val(data):
    """Split the data into 5 splits for cross validation, splitting the data into training, validation and test data. Return a list of tuples with the training, validation and test data."""
    splits = []
    num_splits = 5
    train_data = []
    test_data = []
    val_data = []

    split_train_test_size = len(data) // num_splits
    splits_train_test = []
    for i in range(num_splits):
        if i != num_splits - 1:
            sp = data.iloc[i * split_train_test_size : (i + 1) * split_train_test_size]
        else:
            sp = data.iloc[i * split_train_test_size :]
        splits_train_test.append(sp)
    for i in range(num_splits):
        for j in range(num_splits):
            if j != i:
                test_data = splits_train_test[j]
                val_data = splits_train_test[i]
                train_data = pd.concat(
                    [
                        splits_train_test[k]
                        for k in range(num_splits)
                        if k != i and k != j
                    ]
                )
                splits.append((train_data, val_data, test_data))

    return splits


def write_results_to_csv(
    results_knn,
    results_dt,
    results_nn,
    file_name,
    data_arguments,
    folder,
    neural_network: NeuralNetwork,
    knn_and_dt=True,
):
    """Write the results of the evaluation of the models to a csv file. The results are the accuracy, the confusion matrix, the log loss, the precision, the recall and the F1 score."""
    if knn_and_dt:
        i = 0
        max_accuracy_knn_params = max(
            results_knn, key=lambda x: results_knn[x].accuracy
        )
        max_precision_knn_params = max(
            results_knn, key=lambda x: results_knn[x].precision
        )
        max_recall_knn_params = max(results_knn, key=lambda x: results_knn[x].recall)
        max_f1_score_knn_params = max(
            results_knn, key=lambda x: results_knn[x].f1_score
        )

        max_accuracy_dt_params = max(results_dt, key=lambda x: results_dt[x].accuracy)
        max_precision_dt_params = max(results_dt, key=lambda x: results_dt[x].precision)
        max_recall_dt_params = max(results_dt, key=lambda x: results_dt[x].recall)
        max_f1_score_dt_params = max(results_dt, key=lambda x: results_dt[x].f1_score)

        while os.path.isfile(
            f"results/{folder}/knn/knn_results_{file_name[:-4]}{i}.csv"
        ):
            i += 1
        with open(
            f"results/{folder}/knn/knn_results_{file_name[:-4]}{i}.csv", "w"
        ) as f:
            f.write("DATA COLUMN NAMES: " + str(data_arguments.columns) + "\n")
            f.write(
                "BEST KNN MODEL ACCURACY FOR PARAMS: "
                + str(max_accuracy_knn_params)
                + ". "
                + str(results_knn[max_accuracy_knn_params])
                + "\n"
            )
            f.write(
                "BEST KNN MODEL PRECISION FOR PARAMS: "
                + str(max_precision_knn_params)
                + ". "
                + str(results_knn[max_precision_knn_params])
                + "\n"
            )
            f.write(
                "BEST KNN MODEL RECALL FOR PARAMS: "
                + str(max_recall_knn_params)
                + ". "
                + str(results_knn[max_recall_knn_params])
                + "\n"
            )
            f.write(
                "BEST KNN MODEL F1 SCORE FOR PARAMS: "
                + str(max_f1_score_knn_params)
                + ". "
                + str(results_knn[max_f1_score_knn_params])
                + "\n"
            )
            for key in results_knn.keys():
                f.write("%s,%s\n" % (key, results_knn[key]))
        i = 0
        while os.path.isfile(f"results/{folder}/dt/dt_results_{file_name[:-4]}{i}.csv"):
            i += 1
        with open(f"results/{folder}/dt/dt_results_{file_name[:-4]}{i}.csv", "w") as f:
            f.write("DATA COLUMN NAMES: " + str(data_arguments.columns) + "\n")
            f.write(
                "BEST DT MODEL ACCURACY FOR PARAMS: "
                + str(max_accuracy_dt_params)
                + ". "
                + str(results_dt[max_accuracy_dt_params])
                + "\n"
            )
            f.write(
                "BEST DT MODEL PRECISION FOR PARAMS: "
                + str(max_precision_dt_params)
                + ". "
                + str(results_dt[max_precision_dt_params])
                + "\n"
            )
            f.write(
                "BEST DT MODEL RECALL FOR PARAMS: "
                + str(max_recall_dt_params)
                + ". "
                + str(results_dt[max_recall_dt_params])
                + "\n"
            )
            f.write(
                "BEST DT MODEL F1 SCORE FOR PARAMS: "
                + str(max_f1_score_dt_params)
                + ". "
                + str(results_dt[max_f1_score_dt_params])
                + "\n"
            )
            for key in results_dt.keys():
                f.write("%s,%s\n" % (key, results_dt[key]))

    i = 0
    max_accuracy_nn_params = max(results_nn, key=lambda x: results_nn[x].accuracy)
    max_precision_nn_params = max(results_nn, key=lambda x: results_nn[x].precision)
    max_recall_nn_params = max(results_nn, key=lambda x: results_nn[x].recall)
    max_f1_score_nn_params = max(results_nn, key=lambda x: results_nn[x].f1_score)
    while os.path.isfile(f"results/{folder}/nn/nn_results_{file_name[:-4]}{i}.csv"):
        i += 1
    with open(f"results/{folder}/nn/nn_results_{file_name[:-4]}{i}.csv", "w") as f:
        f.write("DATA COLUMN NAMES: " + str(data_arguments.columns) + "\n")
        f.write(
            f"NN with {neural_network.get_first_layer_units()}, {neural_network.get_second_layer_units()}, {neural_network.get_third_layer_units()} units for hidden layers\n"
        )
        f.write(
            "BEST NN MODEL ACCURACY FOR PARAMS: "
            + str(max_accuracy_nn_params)
            + ". "
            + str(results_nn[max_accuracy_nn_params])
            + "\n"
        )
        f.write(
            "BEST NN MODEL PRECISION FOR PARAMS: "
            + str(max_precision_nn_params)
            + ". "
            + str(results_nn[max_precision_nn_params])
            + "\n"
        )
        f.write(
            "BEST NN MODEL RECALL FOR PARAMS: "
            + str(max_recall_nn_params)
            + ". "
            + str(results_nn[max_recall_nn_params])
            + "\n"
        )
        f.write(
            "BEST NN MODEL F1 SCORE FOR PARAMS: "
            + str(max_f1_score_nn_params)
            + ". "
            + str(results_nn[max_f1_score_nn_params])
            + "\n"
        )

        for key in results_nn.keys():
            f.write("%s,%s\n" % (key, results_nn[key]))


def format_knn_params(model):
    """Format the parameters of the KNN model as a string. Return the string with the parameters of the KNN model."""
    string = (
        "algorithm: "
        + model.get_params()["algorithm"]
        + ", metric: "
        + model.get_params()["metric"]
        + ", weights: "
        + model.get_params()["weights"]
        + ", n_neighbors: "
        + str(model.get_params()["n_neighbors"])
    )
    return string


def format_dt_params(model):
    """Format the parameters of the Decision Tree model as a string. Return the string with the parameters of the Decision Tree model."""
    string = (
        "max_depth: "
        + str(model.get_params()["max_depth"])
        + ", min_samples_split: "
        + str(model.get_params()["min_samples_split"])
        + ", min_samples_leaf: "
        + str(model.get_params()["min_samples_leaf"])
        + ", criterion: "
        + str(model.get_params()["criterion"])
        + ", splitter: "
        + str(model.get_params()["splitter"])
    )
    return string


def get_model(
    neural_network: NeuralNetwork, input_size, reg_weight=0.0, dr=0.0, count_classes=2
):
    """Get the neural network model with the number of units for the hidden layers, the input size, the regularization weight, the dropout rate and the number of classes. Return the neural network model."""
    """Adapted method from nlp_lab2_solution.ipynb"""
    model = Sequential()
    model.add(
        Dense(
            units=neural_network.get_first_layer_units(),
            activation="relu",
            kernel_regularizer=regularizers.l2(reg_weight),
            input_shape=(input_size,),
        )
    )
    model.add(Dropout(dr))

    model.add(
        Dense(
            units=neural_network.get_second_layer_units(),
            activation="relu",
            kernel_regularizer=regularizers.l2(reg_weight),
        )
    )
    model.add(Dropout(dr))

    model.add(
        Dense(
            units=neural_network.get_third_layer_units(),
            activation="relu",
            kernel_regularizer=regularizers.l2(reg_weight),
        )
    )
    model.add(Dropout(dr))
    if count_classes == 2:
        model.add(
            Dense(
                units=1,
                activation="sigmoid",
                kernel_regularizer=regularizers.l2(reg_weight),
            )
        )
    else:
        model.add(
            Dense(
                units=count_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(reg_weight),
            )
        )
    return model


def train_model(data, neural_network: NeuralNetwork, knn_and_dt=True):
    """Train the models KNN, Decision Tree and Neural Network with the data. Return the results of the evaluation of the models."""
    data = shuffle(data)

    # Definir hiperparámetros
    k_values = [
        1,
        5,
        9,
        13,
        17,
        23,
        27,
        31,
        35,
        39,
        45,
        49,
        53,
        57,
        61,
        65,
        69,
        75,
        79,
        83,
        87,
        91,
        95,
    ]
    metrics = [
        "cosine",
        "euclidean",
        "manhattan",
        "minkowski",
        "chebyshev",
        "canberra",
        "braycurtis",
    ]
    weights = ["uniform", "distance"]
    depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 9]
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    criterion_values = ["gini", "entropy"]
    splitter_values = ["best", "random"]

    # Dividir los datos en 5 splits para cross validation (KNN y DT)
    if knn_and_dt:
        splits_knn = cross_validation_splits(data)
    # Dividir los datos en 20 splits para cross validation (NN)
    splits_nn = cross_validation_splits_with_val(data)

    # Definir funciones de pérdida
    count_classes = len(np.unique(data["clase"]))
    if count_classes > 2:
        loss_function_used = "categorical_crossentropy"
    else:
        loss_function_used = "binary_crossentropy"

    results_knn = {}
    results_dt = {}
    results_nn = {}
    models_nn = {}

    if knn_and_dt:
        for train_data, test_data in splits_knn:
            X_train = train_data.drop(columns=["clase"])
            y_train = train_data["clase"]
            X_test = test_data.drop(columns=["clase"])
            y_test = test_data["clase"]

            # MODELO 1: KNN

            for i in k_values:
                for w in weights:
                    for m in metrics:
                        model = KNeighborsClassifier(n_neighbors=i, weights=w, metric=m)
                        model.fit(X_train, y_train)
                        model_format = format_knn_params(model)
                        y_pred = model.predict(X_test)
                        report = classification_report(
                            y_test, y_pred, output_dict=True, digits=6
                        )

                        macro_f1 = report["macro avg"]["f1-score"]
                        results = Results(
                            report["accuracy"],
                            confusion_matrix(y_test, y_pred, normalize="true"),
                            report["macro avg"]["precision"],
                            report["macro avg"]["recall"],
                            macro_f1,
                        )

                        if results_knn.get(model_format):
                            results_knn[model_format].append(results)
                        else:
                            results_knn[model_format] = [results]

            # MODELO 2: ÁRBOL DE DECISIÓN

            for depth in depth_values:
                for min_samples_split in min_samples_split_values:
                    for min_samples_leaf in min_samples_leaf_values:
                        for criterion in criterion_values:
                            for splitter in splitter_values:
                                model = DecisionTreeClassifier(
                                    criterion=criterion,
                                    splitter=splitter,
                                    max_depth=depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                )
                                model.fit(X_train, y_train)
                                model_format = format_dt_params(model)
                                y_pred = model.predict(X_test)

                                report = classification_report(
                                    y_test, y_pred, output_dict=True, digits=6
                                )
                                results = Results(
                                    report["accuracy"],
                                    confusion_matrix(y_test, y_pred, normalize="true"),
                                    report["macro avg"]["precision"],
                                    report["macro avg"]["recall"],
                                    report["macro avg"]["f1-score"],
                                )
                                if results_dt.get(model_format):
                                    results_dt[model_format].append(results)
                                else:
                                    results_dt[model_format] = [results]

    # CASO 3: RED NEURONAL DE VARIAS CAPAS, GRID SEARCH

    # Establecer hiperparámetros
    lrs = [0.0001, 0.001, 0.01]
    drs = [0.0, 0.2, 0.5]
    regs = [0.0, 0.0001]

    num_epochs = 100
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5)

    for train_data, val_data, test_data in splits_nn:

        X_train = train_data.drop(columns=["clase"])
        y_train = train_data["clase"]
        X_val = val_data.drop(columns=["clase"])
        y_val = val_data["clase"]
        X_test = test_data.drop(columns=["clase"])
        y_test = test_data["clase"]

        if count_classes > 2:
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            y_val = to_categorical(y_val)

        input_dim = X_train.shape[1]

        for lr in lrs:
            for reg in regs:
                for dr in drs:
                    model = get_model(neural_network, input_dim, reg, dr, count_classes)

                    model.compile(
                        loss=loss_function_used,
                        optimizer=Adam(learning_rate=lr),
                        metrics=["accuracy"],
                    )
                    model.fit(
                        X_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[early_stop],
                    )
                    acc = model.evaluate(X_val, y_val, verbose=0)[1]
                    y_pred = model.predict(X_test)

                    report = classification_report(
                        y_test.argmax(axis=1),
                        y_pred.argmax(axis=1),
                        output_dict=True,
                    )
                    results = Results(
                        acc,
                        confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)),
                        report["macro avg"]["precision"],
                        report["macro avg"]["recall"],
                        report["macro avg"]["f1-score"],
                    )

                    if results_nn.get("lr{}-reg{}-dr{}".format(lr, reg, dr)):
                        results_nn["lr{}-reg{}-dr{}".format(lr, reg, dr)].append(
                            results
                        )
                        models_nn["lr{}-reg{}-dr{}".format(lr, reg, dr)].append(model)
                    else:
                        results_nn["lr{}-reg{}-dr{}".format(lr, reg, dr)] = [results]
                        models_nn["lr{}-reg{}-dr{}".format(lr, reg, dr)] = [model]

    # Compute mean of results of knn and dt
    if knn_and_dt:
        for key in results_knn:
            results = results_knn[key]
            accuracies = [r.accuracy for r in results]
            mean_accuracy = np.mean(accuracies)
            confusions_matrix = [r.confusion_matrix for r in results]
            try:
                mean_confusion_matrix = np.mean(confusions_matrix, axis=0)
            except Exception as e:
                mean_confusion_matrix = 0
                pass
            precision = [r.precision for r in results]
            mean_precision = np.mean(precision)
            recall = [r.recall for r in results]
            mean_recall = np.mean(recall)
            f1_score = [r.f1_score for r in results]
            mean_f1_score = np.mean(f1_score)
            results_knn[key] = Results(
                mean_accuracy,
                mean_confusion_matrix,
                mean_precision,
                mean_recall,
                mean_f1_score,
            )
        for key in results_dt:
            results = results_dt[key]
            accuracies = [r.accuracy for r in results]
            mean_accuracy = np.mean(accuracies)
            confusions_matrix = [r.confusion_matrix for r in results]
            try:
                mean_confusion_matrix = np.mean(confusions_matrix, axis=0)
            except Exception as e:
                mean_confusion_matrix = 0
                pass
            precision = [r.precision for r in results]
            mean_precision = np.mean(precision)
            recall = [r.recall for r in results]
            mean_recall = np.mean(recall)
            f1_score = [r.f1_score for r in results]
            mean_f1_score = np.mean(f1_score)
            results_dt[key] = Results(
                mean_accuracy,
                mean_confusion_matrix,
                mean_precision,
                mean_recall,
                mean_f1_score,
            )
    # Compute mean of results of nn
    for key in results_nn:
        results = results_nn[key]

        accuracies = [r.accuracy for r in results]
        mean_accuracy = np.mean(accuracies)
        confusions_matrix = [r.confusion_matrix for r in results]
        mean_confusion_matrix = 0
        try:
            mean_confusion_matrix = np.mean(confusions_matrix, axis=0)
        except Exception as e:
            mean_confusion_matrix = 0
            pass
        precision = [r.precision for r in results]
        mean_precision = np.mean(precision)
        recall = [r.recall for r in results]
        mean_recall = np.mean(recall)
        f1_score = [r.f1_score for r in results]
        mean_f1_score = np.mean(f1_score)
        results_nn[key] = Results(
            mean_accuracy,
            mean_confusion_matrix,
            mean_precision,
            mean_recall,
            mean_f1_score,
        )

    return results_knn, results_dt, results_nn


def load_train_write(file_name, folder, drops, neural_network, knn_and_dt=True):
    """Load the features of the corpus, train the models and write the results to a csv file."""
    data_arguments = load_features(file_name)

    data_arguments = pd.DataFrame(data_arguments)
    # append text_position column to drops
    if "text_position" not in drops:
        drops.append("text_position")

    # remove columns
    data_arguments = data_arguments.drop(columns=drops)

    # train models
    results_knn, results_dt, results_nn = train_model(
        data_arguments, neural_network, knn_and_dt
    )

    # store results in csv files
    write_results_to_csv(
        results_knn,
        results_dt,
        results_nn,
        file_name,
        data_arguments,
        folder,
        neural_network,
        knn_and_dt,
    )


def main(number):
    """Main function to train the models and write the results to a csv file."""
    if number == "1":
        file_name = "arguments_features.txt"
        folder = "arguments_results"
    elif number == "2":
        file_name = "arguments_phrases_features.txt"
        folder = "phrases_results"
    elif number == "3":
        file_name = "arguments_premises_claims_features.txt"
        folder = "premises_claims_results"
    elif number == "4":
        file_name = "arguments_coherence_features.txt"
        folder = "coherence_results"
    elif number == "5":
        file_name = "arguments_consistence_features.txt"
        folder = "consistence_results"
    elif number == "6":
        file_name = "arguments_emotional_ethic_features.txt"
        folder = "emotionalethic_results"
    elif number == "7":
        file_name = "arguments_persuasion_features.txt"
        folder = "persuasion_results"
    elif number == "8":
        file_name = "arguments_premise_validation_features.txt"
        folder = "premisevalidation_results"

    neural1 = NeuralNetwork(128, 64, 32)
    neural2 = NeuralNetwork(64, 32, 16)
    neural3 = NeuralNetwork(256, 128, 64)
    load_train_write(
        file_name,
        folder,
        ["bow_unigrams", "bow_bigrams", "bow_trigrams"],
        neural1,
    )

    load_train_write(
        file_name,
        folder,
        ["bow_unigrams", "bow_bigrams", "bow_trigrams"],
        neural2,
        False,
    )

    load_train_write(
        file_name,
        folder,
        ["bow_unigrams", "bow_bigrams", "bow_trigrams"],
        neural3,
        False,
    )


if __name__ == "__main__":
    main("1")
    main("2")
    main("3")
    main("4")
    main("5")
    main("6")
    main("7")
    main("8")
