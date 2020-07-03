import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

def visualize():
    dataset_train = []  # инициализируем пустой список

    # местонахождение файлов и их метки
    dataset_place = {"DATASETS/TRAINING/Music_dataset.csv": 1, "DATASETS/TRAINING/Sport_dataset.csv": 2, "DATASETS/TRAINING/Cinema_dataset.csv": 3,
                     "DATASETS/TRAINING/Makeup_dataset.csv": 4, "DATASETS/TRAINING/Gaming_dataset.csv": 5, "DATASETS/TRAINING/Literature_dataset.csv": 6,
                     "DATASETS/TRAINING/Programming_dataset.csv": 7, "DATASETS/TRAINING/Psychology_dataset.csv": 8, "DATASETS/TRAINING/Cooking_dataset.csv": 9,
                     "DATASETS/TRAINING/Travel_dataset.csv": 10, "DATASETS/TRAINING/Cars_dataset.csv": 11, "DATASETS/TRAINING/Another_dataset.csv": 12}

    def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

        if axes is None:
            _, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].set_title(title)

        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        # Plot txt
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        # Plot txt
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

    class PreProcess(object):

        def __init__(self, dataset_place, label):
            self.dataset_place = dataset_place
            self.label = label

        # векторизация данных
        def vectorizer(self):
            df = pd.DataFrame(pd.read_csv(self.dataset_place))
            [dataset_train.append(np.append(vectorizer.fit_transform([i]).toarray(), self.label)) for i in df['txt'].values.tolist()]

        # открытие словаря статик метод
        @staticmethod
        def open_vocab():
            df = pd.DataFrame(pd.read_csv("DATASETS/vocab.csv"))
            return df['vocab'].values.tolist()

    vectorizer = TfidfVectorizer(vocabulary=PreProcess.open_vocab())  # преобразует текст в цифры Его нужно ОТКРЫТЬ ОДИН РАЗ

    print('\nGENERATING DATA...')
    for key in dataset_place:
        print(key)  # дебаг функция, показывает какой конкретный датасет преобразовывается
        PreProcess(key, dataset_place[key]).vectorizer()  # вызов функции vectorizer в obj
    print('OK.')

    print('\nSHUFFLING DATA...')
    dataset_train = np.array(dataset_train)  # создаем нампи матрицу
    np.random.shuffle(dataset_train)
    print('OK.')

    print('\nCREATING DATASET...')
    X_train = dataset_train[:int(dataset_train.shape[0]), :int(dataset_train.shape[1]) - 2]  # слайсим features
    Y_train = dataset_train[:int(dataset_train.shape[0]), int(dataset_train.shape[1]) - 1]  # слайсим labels
    print('OK.')

    fig, axes = plt.subplots(3, 4, figsize=(10, 15))
    print('\nCREATING PLOTS...')

    # Random Forest
    title = r"Random Forest"
    clf = RandomForestClassifier()
    plot_learning_curve(clf, title, X_train, Y_train, axes=axes[:, 0], ylim=(0.3, 1.01), n_jobs=-1)
    print('RANDOM FOREST OK.')

    # AdaBoost
    title = r"Ada"
    clf = AdaBoostClassifier()
    plot_learning_curve(clf, title, X_train, Y_train, axes=axes[:, 1], ylim=(0.3, 1.01), n_jobs=-1)
    print('ADABOOST OK.')

    # Desicion Tree
    title = r"Desicion Tree"
    clf = DecisionTreeClassifier()
    plot_learning_curve(clf, title, X_train, Y_train, axes=axes[:, 2], ylim=(0.3, 1.01), n_jobs=-1)
    print('DESICION TREE OK.')

    # Random Forest
    title = r"Naive Bayes"
    clf = GaussianNB()
    plot_learning_curve(clf, title, X_train, Y_train, axes=axes[:, 3], ylim=(0.3, 1.01), n_jobs=-1)
    print('Naive Bayes OK.')

    plt.show()
