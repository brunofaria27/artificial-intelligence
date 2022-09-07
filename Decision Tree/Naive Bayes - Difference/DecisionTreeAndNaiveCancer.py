# Importando bibliotecas
import pandas as pd

from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Importar a biblioteca para mostrar a matriz de confusão
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Importando bibliotecas para calcular métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pydot
import pydotplus
import seaborn as sns

from imblearn.over_sampling import SMOTE # Oversampling
from imblearn.under_sampling import RandomUnderSampler # Undersampling

# Ler dados do DataSet
csv_dir_treinamento = "/content/sample_data/breast-cancer.csv"
data = pd.read_csv(csv_dir_treinamento, delimiter=",")

# Tratar dados nominais para númericos (Treinamento)
treinamento_classification = data['Class']
data.drop(["Class"], axis=1, inplace=True)
treinamento_dados = pd.get_dummies(data, columns=['menopause', 'age', 'tumor-size', 'breast-quad', 'inv-nodes'])

# Substitui os valores nulos da coluna "node-caps" pelo valor mais frequente da coluna
data['node-caps'] = data['node-caps'].replace('?', data['node-caps'].mode()[0])

# Atualizar tabela com binários
labelencoder =  preprocessing.LabelEncoder()
treinamento_dados['breast'] = labelencoder.fit_transform(treinamento_dados['breast'])
treinamento_dados['irradiat'] = labelencoder.fit_transform(treinamento_dados['irradiat'])
treinamento_dados['node-caps'] = labelencoder.fit_transform(treinamento_dados['node-caps'])

# Separar dados de treinamento e de testes
dataset_treino, dataset_teste, dataset_treino_class, dataset_teste_class = train_test_split(treinamento_dados, treinamento_classification, test_size=0.20, random_state=0)

# Tratar dataset desbalanceado
smote = SMOTE(random_state=0)
# smote = RandomUnderSampler(random_state=0)
dataset_treino, dataset_treino_class = smote.fit_resample(dataset_treino, dataset_treino_class)
dataset_teste, dataset_teste_class = smote.fit_resample(dataset_teste, dataset_teste_class)

# Treinar árvore
tree_data = DecisionTreeClassifier(criterion="entropy")
tree_data.fit(dataset_treino, dataset_treino_class)

# Mostrar matriz de confusão
conf_matrix = confusion_matrix(dataset_teste_class, tree_data.predict(dataset_teste))
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=dataset_teste_class.values)
cmd.plot(values_format="d")
plt.show()

# Mostrar métricas
print('Árvore')
print("Accuracy score = ", accuracy_score(dataset_teste_class, tree_data.predict(dataset_teste)))
print('\n')
print(classification_report(dataset_teste_class, tree_data.predict(dataset_teste)))
print('\n')

tn, fp, fn, tp = confusion_matrix(dataset_teste_class, tree_data.predict(dataset_teste)).ravel()
tnr = tn / (tn + fp) # true negative rate 
fpr = fp / (tn + fp) # false positive rate 
fnr = fn / (fn + tp) # false negative rate 
tpr = tp / (tp + fn) # true positive rate
print('True negative rate: ' + str(tnr))
print('False positive rate: ' + str(fpr))
print('False negative rate: ' + str(fnr))
print('True positive rate: ' + str(tpr))

# Treinar Naive Bayes
gnb = GaussianNB()
gnb.fit(dataset_treino, dataset_treino_class)

# Mostrar matriz de confusão
conf_matrix = confusion_matrix(dataset_teste_class, gnb.predict(dataset_teste))
cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=dataset_teste_class.values)
cmd.plot(values_format="d")
plt.show()

# Mostrar métricas
print('Naive Bayes')
print("Accuracy score = ", accuracy_score(dataset_teste_class, gnb.predict(dataset_teste)))
print('\n')
print(classification_report(dataset_teste_class, gnb.predict(dataset_teste)))
print('\n')

tn, fp, fn, tp = confusion_matrix(dataset_teste_class, gnb.predict(dataset_teste)).ravel()
tnr = tn / (tn + fp) # true negative rate 
fpr = fp / (tn + fp) # false positive rate 
fnr = fn / (fn + tp) # false negative rate 
tpr = tp / (tp + fn) # true positive rate
print('True negative rate: ' + str(tnr))
print('False positive rate: ' + str(fpr))
print('False negative rate: ' + str(fnr))
print('True positive rate: ' + str(tpr))