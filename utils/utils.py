import torch
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse
import yaml

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    acc_val, acc = svc_classify(x, y, search)
    return acc, acc_val

def arg_parse(DS="MUTAG"):
    parser = argparse.ArgumentParser(description='DGCL Arguments.')

    parser.add_argument('--DS', type=str, default=DS)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_gc_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--aug', type=str)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--drop_ratio', type=float)
    parser.add_argument('--num_latent_factors', type=int)
    parser.add_argument('--head_layers', type=int)
    parser.add_argument('--JK', type=str, choices=['last', 'sum'])
    parser.add_argument('--residual', type=int, choices=[0, 1])
    parser.add_argument('--proj', type=int, choices=[0, 1])
    parser.add_argument('--pool', type=str, choices=['mean', 'sum', 'max'])
    parser.add_argument('--fe', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--device", type=int)

    args, unknown = parser.parse_known_args([])

    with open(f'config/{DS}.yml', 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v is not None:
            config_yaml[k] = v
    config_ns = argparse.Namespace(**config_yaml)

    return config_ns