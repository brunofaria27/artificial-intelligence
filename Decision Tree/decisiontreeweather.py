# -*- coding: utf-8 -*-
# Importando bibliotecas
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

import pydot
import pydotplus

from IPython.display import Image

# Importar a biblioteca para mostrar a matriz de confusão da árvore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Importando bibliotecas para calcular métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Ler dados do DataSet
csv_dir_treinamento = "/content/sample_data/weather.nominal.csv"
data_treino = pd.read_csv(csv_dir_treinamento, delimiter=",")

# Tratar dados nominais para númericos (Treinamento)
treinamento_classification = data_treino['play']
data_treino.drop(["play"], axis=1, inplace=True)
treinamento_dados = pd.get_dummies(data_treino, columns=['outlook', 'temperature', 'humidity'])

# Atualizar tabela com binários
labelencoder =  preprocessing.LabelEncoder()
treinamento_dados['windy'] = labelencoder.fit_transform(treinamento_dados['windy'])

# Separar dados de treinamento e de testes
from sklearn.model_selection import train_test_split
dataset_treino, dataset_teste, dataset_treino_class, dataset_teste_class = train_test_split(treinamento_dados, treinamento_classification, test_size=0.20, random_state=0)

# Criando a árvore e definindo o criterio de criação usando entropia
tree_weather = DecisionTreeClassifier(criterion="entropy")
tree_weather = tree_weather.fit(dataset_treino, dataset_treino_class)

dot_data = tree.export_graphviz(tree_weather, out_file=None,
                                feature_names=dataset_treino.columns.values,
                                class_names=treinamento_classification.unique(),
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data)
graph[0].write_png('tree_weather.png')
Image(filename='tree_weather.png')

# Mostrar a matriz de confusão
conf_matrix = confusion_matrix(dataset_teste_class, tree_weather.predict(dataset_teste))
labels = ["Vai jogar", "Não vai jogar"]
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
cmd.plot(values_format="d")
plt.show()

# Mostrar métricas
print("Accuracy score = ", accuracy_score(dataset_teste_class, tree_weather.predict(dataset_teste)))
print('\n')
print(classification_report(dataset_teste_class, tree_weather.predict(dataset_teste), target_names=labels))
print('\n')
tn, fp, fn, tp = confusion_matrix(dataset_teste_class, tree_weather.predict(dataset_teste)).ravel()
tnr = tn / (tn + fp) # true negative rate 
fpr = fp / (tn + fp) # false positive rate 
fnr = fn / (fn + tp) # false negative rate 
tpr = tp / (tp + fn) # true positive rate
print('True negative rate: ' + str(tnr))
print('False positive rate: ' + str(fpr))
print('False negative rate: ' + str(fnr))
print('True positive rate: ' + str(tpr))