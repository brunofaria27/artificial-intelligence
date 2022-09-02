# Importando bibliotecas
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

import pydot
import pydotplus

# Importando bibliotecas para calcular métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importar a biblioteca para mostrar a matriz de confusão da árvore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from IPython.display import Image

# Ler dados do DataSet
csv_dir_treinamento = "/content/sample_data/geral.csv"
data_treino = pd.read_csv(csv_dir_treinamento, delimiter=";")

# Tratar dados nominais para númericos (Treinamento)
treinamento_classification = data_treino['conc']
data_treino.drop(["conc"], axis=1, inplace=True)
data_treino.drop(["Exemplo"], axis=1, inplace=True)
treinamento_dados = pd.get_dummies(data_treino, columns=['Cliente', 'Preço', 'Tipo', 'Tempo'])

# Atualizar tabela com binários
labelencoder =  preprocessing.LabelEncoder()
treinamento_dados['Alternativo'] = labelencoder.fit_transform(treinamento_dados['Alternativo'])
treinamento_dados['Bar'] = labelencoder.fit_transform(treinamento_dados['Bar'])
treinamento_dados['Sex/Sab'] = labelencoder.fit_transform(treinamento_dados['Sex/Sab'])
treinamento_dados['fome'] = labelencoder.fit_transform(treinamento_dados['fome'])
treinamento_dados['Chuva'] = labelencoder.fit_transform(treinamento_dados['Chuva'])
treinamento_dados['Res'] = labelencoder.fit_transform(treinamento_dados['Res'])

# Separar dados de treinamento e de testes
from sklearn.model_selection import train_test_split
dataset_treino, dataset_teste, dataset_treino_class, dataset_teste_class = train_test_split(treinamento_dados, treinamento_classification, test_size=0.20, random_state=0)

# Criando a árvore e definindo o criterio de criação usando entropia
tree_restaurant = DecisionTreeClassifier(criterion="entropy")
tree_restaurant = tree_restaurant.fit(dataset_treino, dataset_treino_class)

# Printar a árvore
dot_data = tree.export_graphviz(tree_restaurant, out_file=None,
                                feature_names=dataset_treino.columns.values,
                                class_names=treinamento_classification.unique(),
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data)
graph[0].write_png('tree_restaurant.png')
Image(filename='tree_restaurant.png')

# Gerar e mostrar matriz de confusão
conf_matrix = confusion_matrix(dataset_teste_class, tree_restaurant.predict(dataset_teste))
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=dataset_teste_class.values)
cmd.plot(values_format="d")
plt.show()

# Mostrar métricas
print("Accuracy score = ", accuracy_score(dataset_teste_class, tree_restaurant.predict(dataset_teste)))
print('\n')
print(classification_report(dataset_teste_class, tree_restaurant.predict(dataset_teste)))
print('\n')

tn, fp, fn, tp = confusion_matrix(dataset_teste_class, tree_restaurant.predict(dataset_teste)).ravel()
tnr = tn / (tn + fp) # true negative rate 
fpr = fp / (tn + fp) # false positive rate 
fnr = fn / (fn + tp) # false negative rate 
tpr = tp / (tp + fn) # true positive rate
print('True negative rate: ' + str(tnr))
print('False positive rate: ' + str(fpr))
print('False negative rate: ' + str(fnr))
print('True positive rate: ' + str(tpr))
