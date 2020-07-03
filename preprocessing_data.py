import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
dataset_train = []  # инициализируем пустой список

# местонахождение файлов и их метки
dataset_place = {"DATASETS/TRAINING/Music_dataset.csv": 1, "DATASETS/TRAINING/Sport_dataset.csv": 2, "DATASETS/TRAINING/Cinema_dataset.csv": 3,
                 "DATASETS/TRAINING/Makeup_dataset.csv": 4, "DATASETS/TRAINING/Gaming_dataset.csv": 5, "DATASETS/TRAINING/Literature_dataset.csv": 6,
                 "DATASETS/TRAINING/Programming_dataset.csv": 7, "DATASETS/TRAINING/Psychology_dataset.csv": 8, "DATASETS/TRAINING/Cooking_dataset.csv": 9,
                 "DATASETS/TRAINING/Travel_dataset.csv": 10, "DATASETS/TRAINING/Cars_dataset.csv": 11, "DATASETS/TRAINING/Another_dataset.csv": 12}


###################################################
# Класс PreProcess
###################################################


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


###################################################

vectorizer = TfidfVectorizer(vocabulary=PreProcess.open_vocab())  # преобразует текст в цифры Его нужно ОТКРЫТЬ ОДИН РАЗ

print('\nGENERATING DATA...')

for key in dataset_place:
    print(key)  # дебаг функция, показывает какой конкретный датасет преобразовывается
    PreProcess(key, dataset_place[key]).vectorizer()  # вызов функции vectorizer в obj

dataset_train = np.array(dataset_train)  # создаем нампи матрицу
np.random.shuffle(dataset_train)

X_train = dataset_train[:int(dataset_train.shape[0]), :int(dataset_train.shape[1]) - 2]  # слайсим features
Y_train = dataset_train[:int(dataset_train.shape[0]), int(dataset_train.shape[1]) - 1]  # слайсим labels

print('\nTRAINING DATA...')
clf = GaussianNB().fit(X_train, Y_train)
print('\nOK.')

print('\nPREPARE TO .ML...')
with open('DATASETS/ML_core.ml', 'wb') as f:
    pickle.dump(clf, f)  # дампим данные в формат PPD

print('\nREADY TO USE.')  # сообщает о готовности