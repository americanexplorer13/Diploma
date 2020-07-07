# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import visualize
import numpy as np
import threading
import pickle
import ast
import re
import os


dataset_place = {"DATASETS/TESTING/VK/Vk_dataset.csv": 0}
dataset_test = []

class ml_core(object):
    def __init__(self, dataset_learn, label):
        self.dataset_place = dataset_learn
        self.label = label

    def vectorizer(self):
        vectorizer = TfidfVectorizer(vocabulary=ml_core.open_vocab())
        df = pd.DataFrame(pd.read_csv(self.dataset_place))
        [dataset_test.append(np.append(vectorizer.fit_transform([i]).toarray(), self.label)) for i in df['txt'].values.tolist()]

    @staticmethod
    def open_vocab():
        df = pd.DataFrame(pd.read_csv("DATASETS/vocab.csv"))
        return df['vocab'].values.tolist()

    def calc(self, ml):
        calc_result = [0]*12

        for name_ml in os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/DATASETS/"):
            if re.search(r"{}".format(ml), name_ml):
                with open('DATASETS/' + ml, 'rb') as f:
                    clf = pickle.load(f)
                break

        for key in dataset_place:  # открывает файл vk_dataset и преображает текст в цифры
            ml_core(key, dataset_place[key]).vectorizer()

        dataset_answer = np.array(dataset_test)

        limitY = int(dataset_answer.shape[1]) - 2  # нужно чтобы он отделял данные от меток
        X_pred = clf.predict(dataset_answer[:, :limitY])

        for i in X_pred:
            calc_result[int(i) - 1] += 1

        z = 0

        for i in calc_result:
            if i > 0:
                calc_result[z] = (i / (len(X_pred) / 5)) * (np.log(len(X_pred) / i))  # формула TF-IDF.
                z += 1
            else:
                calc_result[z] = 0.0  # Формула работает только если ее значение != 0 иначе у нее случается сбой.
                z += 1

        return str(calc_result)


class Ui_MainWindow(object):

    def ML_button(self, ml_engine, frame):
        ml_buttons = {1: self.textEdit.setText, 2: self.textEdit_2.setText, 3: self.textEdit_3.setText, 4: self.textEdit_4.setText}
        d = threading.Thread(target=ml_buttons[frame](ml_core.calc(self, ml="{}.ml".format(ml_engine))))
        d.start()

    def save(self):
        textEditFrame = {"Random_forest": self.textEdit, "AdaBoost": self.textEdit_2, "Desicion_Tree": self.textEdit_3, "Naive": self.textEdit_4}
        to_save = []

        for key in textEditFrame.keys():
            try:
                t = ast.literal_eval(textEditFrame[key].toPlainText())
            except SyntaxError:
                t = [0]*12
            to_save.append((key, t))

        with connection.cursor() as cursor:
            sql = "INSERT INTO saved_results(saved_results1ID, result) VALUES (%s, %s)"
            cursor.execute(sql, (None, str(to_save)))

        filter = "csv (*.csv)"
        from_App = QFileDialog.getSaveFileName(caption="Сохранить результаты моделей", filter=filter)
        try:
            pd.DataFrame.from_dict(dict(to_save)).to_csv(from_App[0], index=False)
        except FileNotFoundError:
            return -1

    @staticmethod
    def GetFromSQL():
        with connection.cursor() as cursor:
            sql = "SELECT * FROM saved_results"
            cursor.execute(sql)
            result = cursor.fetchone()

        filter = "csv (*.csv)"
        from_SQL = QFileDialog.getSaveFileName(caption="Загрузить данные с БД", filter=filter)
        try:
            pd.DataFrame({'index': {'saved_results1ID': result['saved_results1ID']}, 'sav': {'result': result['result']}}).to_csv(from_SQL[0])
        except FileNotFoundError:
            return -1

    @staticmethod
    def show_plot():
        d = threading.Thread(target=visualize.visualize)
        d.start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(768, 550)

        MainWindow.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        MainWindow.setSizePolicy(sizePolicy)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(40, 120, 151, 101))
        self.textEdit.setObjectName("textEdit")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 240, 151, 51))
        self.pushButton.released.connect(lambda: self.ML_button('Random', 1))
        self.pushButton.setStyleSheet("QPushButton {\n"
                                      "background-color: #4CAF50;\n"
                                      "font-weight: bold;\n"
                                      "border: none;\n"
                                      "color: white;\n"
                                      "padding: 15px 32px;\n"
                                      "text-align: center;\n"
                                      "text-decoration: none;\n"
                                      "display: inline-block;\n"
                                      "font-size: 11px;\n"
                                      "border-radius: 10px;\n"
                                      "}\n" 
                                      "QPushButton:hover {background-color: #409243}\n"
                                      "QPushButton:pressed {background-color: #295e2b}\n")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(220, 240, 151, 51))
        self.pushButton_2.released.connect(lambda: self.ML_button('AdaBoost', 2))
        self.pushButton_2.setStyleSheet("QPushButton {\n"
                                        "background-color: #4CAF50;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 11px;\n"
                                        "border-radius: 10px;}\n"
                                        "QPushButton:hover {background-color: #409243}\n"
                                        "QPushButton:pressed {background-color: #295e2b}\n")
        self.pushButton_2.setObjectName("pushButton_2")

        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(220, 120, 151, 101))
        self.textEdit_2.setObjectName("textEdit_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(220, 440, 151, 51))
        self.pushButton_3.released.connect(lambda: self.ML_button('Naive', 4))
        self.pushButton_3.setStyleSheet("QPushButton {\n"
                                        "background-color: #4CAF50;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 11px;\n"
                                        "border-radius: 10px;}\n"
                                        "QPushButton:hover {background-color: #409243}\n"
                                        "QPushButton:pressed {background-color: #295e2b}\n")
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(40, 440, 151, 51))
        self.pushButton_4.released.connect(lambda: self.ML_button('Desicion', 3))
        self.pushButton_4.setStyleSheet("QPushButton {\n"
                                        "background-color: #4CAF50;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 11px;\n"
                                        "border-radius: 10px;}\n"
                                        "QPushButton:hover {background-color: #409243}\n"
                                        "QPushButton:pressed {background-color: #295e2b}\n")
        self.pushButton_4.setObjectName("pushButton_4")

        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(40, 320, 151, 101))
        self.textEdit_3.setObjectName("textEdit_3")

        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(220, 320, 151, 101))
        self.textEdit_4.setObjectName("textEdit_4")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(220, 20, 321, 51))
        self.pushButton_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_5.setStyleSheet("background-color: #3375d6; /* Green */\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 13px;\n"
                                        "border-radius: 10px;")
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.released.connect(self.show_plot)
        self.pushButton_6.setGeometry(QtCore.QRect(400, 120, 151, 51))
        self.pushButton_6.setStyleSheet("QPushButton {\n"
                                        "background-color: #f5a72a;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 13px;\n"
                                        "border-radius: 10px;\n"
                                        "}\n"
                                        "QPushButton:hover {background-color: #d18e24}\n"
                                        "QPushButton:pressed {background-color: #9e6c1c}\n")
        self.pushButton_6.setObjectName("pushButton_6")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 570, 211, 21))
        self.label.setObjectName("label")

        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(400, 190, 151, 51))
        self.pushButton_7.released.connect(self.save)
        self.pushButton_7.setStyleSheet("QPushButton {\n"
                                        "background-color: #9942db;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 13px;\n"
                                        "border-radius: 10px;\n"
                                        "}\n"
                                        "QPushButton:hover {background-color: #7d37b3}\n"
                                        "QPushButton:pressed {background-color: #522475}\n")
        self.pushButton_7.setObjectName("pushButton_7")

        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(400, 260, 151, 51))
        self.pushButton_8.released.connect(self.GetFromSQL)
        self.pushButton_8.setStyleSheet("QPushButton {\n"
                                        "background-color: #ff0000;\n"
                                        "font-weight: bold;\n"
                                        "border: none;\n"
                                        "color: white;\n"
                                        "padding: 15px 32px;\n"
                                        "text-align: center;\n"
                                        "text-decoration: none;\n"
                                        "display: inline-block;\n"
                                        "font-size: 13px;\n"
                                        "border-radius: 10px;\n"
                                        "}\n"
                                        "QPushButton:hover {background-color: #b70000}\n"
                                        "QPushButton:pressed {background-color: #770000}\n")
        self.pushButton_8.setObjectName("pushButton_8")

        MainWindow.setCentralWidget(self.centralwidget)
        self.actionMain = QtWidgets.QAction(MainWindow)
        self.actionMain.setObjectName("actionMain")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ML Interface"))

        self.pushButton.setText(_translate("MainWindow", "Random Forest"))
        self.pushButton_2.setText(_translate("MainWindow", "AdaBoost"))
        self.pushButton_3.setText(_translate("MainWindow", "Naive Bayes"))
        self.pushButton_4.setText(_translate("MainWindow", "Desicion Tree"))

        self.pushButton_5.setText(_translate("MainWindow", "Machine Learning Interface"))
        self.pushButton_6.setText(_translate("MainWindow", "Show Plot"))
        self.pushButton_7.setText(_translate("MainWindow", "Save as..."))
        self.pushButton_8.setText(_translate("MainWindow", "Get from SQL"))

        self.label.setText(_translate("MainWindow", "Powered by PyQt5 | Created by Lunev Kirill"))
        self.actionMain.setText(_translate("MainWindow", "Main"))


if __name__ == "__main__":
    import sys
    import pymysql
    connection = pymysql.connect(host='', user='', password='', db='', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
