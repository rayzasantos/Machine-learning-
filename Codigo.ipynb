# Machine  learning  do banco de dados  do pinguim do Arquipélago Palmer (Antártica)

# Análisando o banco de dados

# Pacotes necessários 
import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt2 
%matplotlib inline


import warnings
warnings.filterwarnings("ignore")


# Entrando com o dataset
df = pd.read_csv("C:/Users/rayza/OneDrive/Área de Trabalho/Topicos/penguins_size.csv")
df.head()

# Informações do dataset
df.info()

# Análise descritiva 
df.describe(include='all')

# Quantidade de observação e variáveis 
print(df.shape)

# Verificando dados ausentes 
df.isnull()

# Soma de cada dados ausentes em cada coluna 
df.isnull().sum()

# Codificando valores de strings para númericas para as colunas "island" e "sex", de modo que, Biscoe = 0, Torgersen = 1 e Dream = 2 e MALE= 0 e FEMALE = 1

#Tranformando island(ilha) em númerica 
df.loc[df["island"] == "Biscoe", "island"] = 0
df.loc[df["island"] == "Torgersen", "island"] = 1
df.loc[df["island"] == "Dream", "island"] = 2
df["island"] = df["island"].astype(float)

# Critério de removação dos ausentes.

def missing_values_table(df):
        # Valores totais ausentes
        mis_val = df.isnull().sum()
        
        # Porcentagem de valores ausentes
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Fazendo  uma tabela com os resultados
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Renomando as colunas
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Classifique a tabela por porcentagem de descendentes ausentes
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Imprima algumas informações resumidas
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Retorne o dataframe com informações ausentes
        return mis_val_table_ren_columns

# Verificando as porcentagens dos valores ausentes de cada variável.
missing= missing_values_table(df)
missing

#  Preenchendo os valores ausentes pela média ou mediana 

from sklearn.impute import SimpleImputer
# definindo a estratégia como 'mais frequente' para imputar pela média
imputer = SimpleImputer(strategy='most_frequent')# estratégia também pode ser média ou mediana 
df.iloc[:,:] = imputer.fit_transform(df)

# verificando as informações novamente.
df.info()

# Verificando se tem dados ausentes 
df.isnull().sum()

# Categorizando a spercies 
categorical = (df.dtypes == "object")
categorical_list = list(categorical[categorical].index)

print(categorical_list)

# Verificando a quantidade do sexo de cada species 
df['sex'].value_counts().plot(kind='bar')

# Verificando a quantidade de cada species 
df['species'].value_counts().plot(kind='bar')

# Matrix de correlação 
plt.figure(figsize=(12,8)) 
sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths = 2)
plt.show()



# Definindo dos float64 em númerico 
numerical_float64 = (df.dtypes == "float64")
numerical_float64_list = list(numerical_float64[numerical_float64].index)

print(numerical_float64_list)

# As 5 primeiras observações dos dados já tratados. 
df.head()

# Treinamento e Teste para os modelos 

# Definindo o x e y 
X = df.iloc[:,1:6].values 
y = df.iloc[:,6:].values

# Defindo o trainamendo 80 e 20 para o teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.8, random_state = 42, shuffle = True ) 


# A forma do traindo e teste 
print(X_train.shape)
print(y_test.shape)

#  Modelo RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators= 500, max_depth= 6)
forest.fit(X_train, y_train) #ajuste o randomforesta dos dados de treinamento
forest_preds = forest.predict(X_test)# fazer previsões fora dos dados de teste
forest.score(X_train, y_train)

# Verificando a eficácia do modelo
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, forest_preds))


# Gráfico da matriz de confunsão
plt.figure(figsize=(8,8)) 

labels = ['Macho', 'Fêmea']

con_matrix = confusion_matrix(y_test, forest_preds)
sns.heatmap(con_matrix, cmap='YlGnBu', annot=True, cbar=False, square=True, 
           xticklabels=labels, 
           yticklabels=labels)

plt.yticks(rotation=0, fontsize=12)
plt.xticks(fontsize=12)

plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Random Forest Classifier Confusion Matrix', fontsize=20)
plt.show()


# Modelo DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
Deci = DecisionTreeClassifier()
Deci = DecisionTreeClassifier(max_depth = 9, min_samples_split = 51)
Deci.fit(X_train, y_train)
y_pred = Deci.predict(X_test)
Deci.score(X_train, y_train)


# Verificando a eficácia do modelo
print(classification_report(y_test, y_pred))

# Gráfico de matriz de confunsão 
plt.figure(figsize=(8,8)) 

labels = ['Macho', 'Fêmea']

con_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(con_matrix, cmap='YlGnBu', annot=True, cbar=False, square=True, 
           xticklabels=labels, 
           yticklabels=labels)

plt.yticks(rotation=0, fontsize=15)
plt.xticks(fontsize=15)

plt.xlabel('Predicted Label', fontsize=15)
plt.ylabel('True Label', fontsize=15)
plt.title('Decision Tree Classifier Confusion Matrix', fontsize=19)
plt.show()

# Modelo Naive Bayes (GaussianNB)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
gaussian.score(X_train, y_train)


# Verificando a eficácia do modelo
print(classification_report(y_test, Y_pred))

# Gráfico de matriz de confunsão 
plt.figure(figsize=(8,8)) 

labels = ['Macho', 'Fêmea']

con_matrix = confusion_matrix(y_test, Y_pred)
sns.heatmap(con_matrix, cmap='YlGnBu', annot=True, cbar=False, square=True, 
           xticklabels=labels, 
           yticklabels=labels)

plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18)

plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.title(' GaussianNB Confusion Matrix', fontsize=18)
plt.show()

# Modelo  LogisticRegression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
reg_pred = logreg.predict(X_test)
logreg.score(X_train, y_train)

# Verificando a eficácia do modelo 
print(classification_report(y_test, reg_pred))

# Gráfico da matriz de confunsão 
plt.figure(figsize=(8,8)) 

labels = ['Macho', 'Fêmea']

con_matrix = confusion_matrix(y_test, reg_pred)
sns.heatmap(con_matrix, cmap='YlGnBu', annot=True, cbar=False, square=True, 
           xticklabels=labels, 
           yticklabels=labels)

plt.yticks(rotation=0, fontsize=15)
plt.xticks(fontsize=15)

plt.xlabel('Predicted Label', fontsize=15)
plt.ylabel('True Label', fontsize=15)
plt.title('Decision Tree Classifier Confusion Matrix', fontsize=19)
plt.show()


## Cálculo do modelo da curva Roc

# Probabilidades de previsão
r_probs = [0 for _ in range(len(y_test))]
rf_probs = forest.predict_proba(X_test)
dt_probs = Deci.predict_proba(X_test)
nb_probs = gaussian.predict_proba(X_test)
rl_probs =logreg.predict_proba(X_test)

# As probabilidades para o resultado positivo são mantidas.
rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]
rl_probs = rl_probs[:, 1]
dt_probs = dt_probs[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score

# Característica operacional do receptor AUROC é a área sob a curva ROC
r_auc = roc_auc_score(y_test, r_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
nb_auc = roc_auc_score(y_test, nb_probs)
rl_auc = roc_auc_score(y_test, rl_probs)
dt_auc = roc_auc_score(y_test, dt_probs)

print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))
print('LogisticRegression: AUROC = %.3f' % (rl_auc))
print('DecisionTreeClassifier: AUROC = %.3f' % (dt_auc))

# Cálculo da curva ROC
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
rl_fpr, rl_tpr, _ = roc_curve(y_test, rl_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)


# Gráfico da curva Roc
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='DecisionTreeClassifier (AUROC = %0.3f)' % dt_auc)
plt.plot(rl_fpr, rl_tpr, marker='.', label='LogisticRegression (AUROC = %0.3f)' % rl_auc)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)
plt.title('ROC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()











