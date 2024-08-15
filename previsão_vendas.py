import numpy as np      # np.arrays
import pandas as pd     # dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# Gráficos
import matplotlib.pyplot as plt  
import matplotlib as mpl
import seaborn as sns

# Datas-tempos
import math
from datetime import datetime
from datetime import timedelta

# Outros pacotes se precisar
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
!pip install pmdarima
import pmdarima as pm
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


import warnings
warnings.filterwarnings("ignore")


pd.options.display.max_columns=100           # ver colunas

df_Loja = pd.read_csv('stores.csv')         # dados das lojas Walmart
df_Loja.rename(columns={'Type':'Tipo'},inplace=True) # Renomeia coluna
df_Loja.rename(columns={'Size':'Tamanho'},inplace=True) # Renomeia coluna
df_Loja.rename(columns={'Store':'Loja'},inplace=True) # Renomeia coluna
df_Loja.head()

df_train = pd.read_csv('train.csv')          # dados de treino
df_train.rename(columns={'Store':'Loja'},inplace=True) # Renomeia coluna
df_train.rename(columns={'Dept':'Depart'},inplace=True) # Renomeia coluna
df_train.rename(columns={'Date':'Data'},inplace=True) # Renomeia coluna
df_train.rename(columns={'Weekly_Sales':'Venda_Semanal'},inplace=True) # Renomeia coluna
df_train.rename(columns={'IsHoliday':'Feriado'},inplace=True) # Renomeia coluna
df_train.head()

df_features = pd.read_csv('features.csv')    # informações externas
df_features.rename(columns={'Store':'Loja'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Date':'Data'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Temperature':'Temperatura'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Fuel_Price':'Preco_gasolina'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Temperature':'Temperatura'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Fuel_Price':'Preco_gasolina'},inplace=True) # Renomeia coluna
df_features.rename(columns={'MarkDown1':'Desconto1'},inplace=True) # Renomeia coluna
df_features.rename(columns={'MarkDown2':'Desconto2'},inplace=True) # Renomeia coluna
df_features.rename(columns={'MarkDown3':'Desconto3'},inplace=True) # Renomeia coluna
df_features.rename(columns={'MarkDown4':'Desconto4'},inplace=True) # Renomeia coluna
df_features.rename(columns={'MarkDown5':'Desconto5'},inplace=True) # Renomeia coluna
df_features.rename(columns={'CPI':'IPC'},inplace=True) # Renomeia coluna
df_features.rename(columns={'Unemployment':'Desemprego'},inplace=True) # Renomeia coluna
df_features.rename(columns={'IsHoliday':'Feriado'},inplace=True) # Renomeia coluna
df_features.head()

df_Loja.head()

df_train.head()

# Juntar 3 datasets utilizando inner join (junção de pelos comuns)
df = df_train.merge(df_features, on=['Loja', 'Data'], how='inner').merge(df_Loja, on=['Loja'], how='inner')
df.head(5)

df.drop(['Feriado_y'], axis=1,inplace=True) # Remove coluna duplicada
df.rename(columns={'Feriado_x':'Feriado'},inplace=True) # Renomeia coluna
df.head() # last ready data set

df.shape	# informa (linhas, colunas)



######### LIMPEZA DE DADOS: Análise de Lojas e Departamentos #########
df['Loja'].nunique() # conta os valores únicos de Loja: 45
df['Depart'].nunique() # conta os valores únicos de Depart: 81


## Crio tabela pivot que me indica as vendas médias por loja e departamento
Loja_Depart_table = pd.pivot_table(df, index='Loja', columns='Depart',
                                  values='Venda_Semanal', aggfunc=np.mean)
display(Loja_Depart_table)

## a tabela indicou que os departamentos vão de 1 a 99 (???)
# missings entre os departamentos, valores negativos de vendas
# a seguir, eu filtro os valores de vendas semanais menores ou iguais a 0

df.loc[df['Venda_Semanal']<=0]	# seleciona as linhas com valores semanais menores que 0

x = (1358*100)/421570     ### Calculo o percentual de valores errados
print(x)
# 0,322129...% valor muito baixo, então posso descartar essas linhas

df = df.loc[df['Venda_Semanal'] > 0]
df.shape # Novo formato da tabela



######### Análise de Datas #########
pd.concat([df['Data'].head(5), df['Data'].tail(5)]) # ver primeiros e últimos 5
## De 05/02/2010 até 26/10/2012


## Coluna Feriado ##
sns.barplot(x='Feriado', y='Venda_Semanal', hue = 'Feriado', data=df, legend=False)	# Gráfico de barras, feriados por vendas semanais, cores diferentes por barra (hue), sem legendas

df_feriado = df.loc[df['Feriado']==True]
df_feriado['Data'].unique()		# Crio arranjo de datas únicas, feriados

df_nao_feriado = df.loc[df['Feriado']==False]
df_nao_feriado['Data'].nunique()	# Conto o números de datas não-únicas

## Todos os feriados (americanos) não estão nos dados. Existem 4 feriados como:
#Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
#Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
#Ação_de_Graças: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
#Natal: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

#Feriados pós 7/09/12 estão no dateset Test para predição.
#Quando olhamos para a informação, vendas médias semanais em feriados são bem maiores do que em não-feriados.
#Nos dados de treino (Train), existem 133 semanas para não-feriados e 10 para feriados.

#Caso queiramos ver diferenças entre os tipos de feriados, precisamos de outras colunas.
#Crio nova coluna para os 4 tipos de feriados e coloco valores boolean (True, False)
#Se a data pertence a feriado ou não

# Datas do Super bowl em set de treino
df.loc[(df['Data'] == '2010-02-12')|(df['Data'] == '2011-02-11')|(df['Data'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Data'] != '2010-02-12')&(df['Data'] != '2011-02-11')&(df['Data'] != '2012-02-10'),'Super_Bowl'] = False

# Datas do Trabalhador em set de treino
df.loc[(df['Data'] == '2010-09-10')|(df['Data'] == '2011-09-09')|(df['Data'] == '2012-09-07'),'Trabalhador'] = True
df.loc[(df['Data'] != '2010-09-10')&(df['Data'] != '2011-09-09')&(df['Data'] != '2012-09-07'),'Trabalhador'] = False

# Datas de Ação de Graças em set de treino
df.loc[(df['Data'] == '2010-11-26')|(df['Data'] == '2011-11-25'),'Ação_de_Graças'] = True
df.loc[(df['Data'] != '2010-11-26')&(df['Data'] != '2011-11-25'),'Ação_de_Graças'] = False

# Datas de Natal em set de treino
df.loc[(df['Data'] == '2010-12-31')|(df['Data'] == '2011-12-30'),'Natal'] = True
df.loc[(df['Data'] != '2010-12-31')&(df['Data'] != '2011-12-30'),'Natal'] = False

# Gráficos
sns.barplot(x='Natal', y='Venda_Semanal', hue = 'Natal', data=df, legend=False) # Feriado Natal vs Não-Natal

sns.barplot(x='Ação_de_Graças', y='Venda_Semanal', hue = 'Ação_de_Graças', data=df, legend=False) # Feriado Ação de Graças vs Não-Ação de Graças

sns.barplot(x='Super_Bowl', y='Venda_Semanal', hue = 'Super_Bowl', data=df, legend=False) # Feriado Super_Bowl vs Não-Super_Bowl

sns.barplot(x='Trabalhador', y='Venda_Semanal', hue = 'Trabalhador', data=df, legend=False) # Feriado Trabalhador vs Não-Trabalhador

#Pelos gráficos, Dia do Trabalhador e Natal não necessariamente aumentam vendas semanais.
#Existe efeito positivo em vendas no Super Bowl, mas a maior diferença é na Ação de Graças (pode ser explicado pela Black Friday).

## A seguir a análise de data por tipo de loja ##
df.groupby(['Natal','Tipo'])['Venda_Semanal'].mean()  # Média de vendas semanal para tipos de lojas no Natal

df.groupby(['Trabalhador','Tipo'])['Venda_Semanal'].mean()  # Média de vendas semanal para tipos de lojas no Dia do Trabalhador

df.groupby(['Ação_de_Graças','Tipo'])['Venda_Semanal'].mean()  # Média de vendas semanal para tipos de lojas na Ação de Graças

df.groupby(['Super_Bowl','Tipo'])['Venda_Semanal'].mean()  # Média de vendas semanal para tipos de lojas no Super Bowl

# Em geral, gráficos de pizza são difíceis de analisar.
# Porém, as diferenças são grandes, então podemos utilizar
# Gráfico em pizza dos percentuais
my_data = [48.88, 37.77 , 13.33 ]  # percentuais
my_labels = 'Tipo A','Tipo B', 'Tipo C' # Títulos
plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) # gráfico em pizza com títulos maiores
plt.axis('equal')
mpl.rcParams.update({'font.size': 20}) # Maiores percentuais

plt.show()

# Quase metade das vendas semanais em feriados vem de A
# Médias semanais por feriado ou não
df.groupby('Feriado')['Venda_Semanal'].mean()

## Gráfico de vendas semanais médias de acordo com feriados por tipos
plt.style.use('seaborn-poster')
labels = ['Ação_de_Graças', 'Super_Bowl', 'Trabalhador', 'Natal']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56,10179.27,9893.45,8031.52]

x = np.arange(len(labels))  # localização dos rótulos
width = 0.25  # espessura das barras

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Tipo A')
rects2 = ax.bar(x , B_means, width, label='Tipo B')
rects3 = ax.bar(x + width, C_means, width, label='Tipo C')

# Adicionar texto para rótulos, título e rótulo do eixo X, etc.
ax.set_ylabel('Média de Vendas Semanais')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos verticais offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30,color='r') # linha média feriados
plt.axhline(y=15952.82,color='green') # linha média não-feriados

fig.tight_layout()

plt.show()

# É mostrado no gráfico que a maior média de vendas é na semana de Ação de Graças entre feriados.
# E, para todos feriados lojas tipo A têm maiores vendas
df.sort_values(by='Venda_Semanal',ascending=False).head(5)

# Top 5 vendas semanais todas em Ação de Graças

## Tamanho e Tipo de loja
df_Loja.groupby('Tipo').describe()['Tamanho'].round(2) # Ver relação tamanho-tipo

plt.figure(figsize=(10,8)) # Ver relação tamanho-tipo
fig = sns.boxplot(x='Tipo', y='Tamanho', data=df, showfliers=False)

# Tamanho do tipo de loja é consistente com vendas. Maior o tamanho, maiores as vendas.
# Walmart classifica lojas de acordo com seus tamanhos de acordo com o gráfico.
# Após o menor tamanho do tipo A, vem o tipo B, e assim sucessivamente.

### Colunas Descontos
# Walmart criou a coluna descontos para ver o efeito nas vendas.
# Muitos missings nessas colunas. Substituo por 0, pois descontos sempre têm valores.
df.isna().sum()

df = df.fillna(0) # Substitui missings por 0
df.isna().sum() # Última checagem de missings

df.describe() # ver dados estatísticos


######### Análise de Vendas #########
x = df['Depart']
y = df['Venda_Semanal']
plt.figure(figsize=(15,5))
plt.title('Venda Semanal por Departamento')
plt.xlabel('Departamentos')
plt.ylabel('Venda Semanal')
plt.scatter(x,y)
plt.show()


plt.figure(figsize=(30,10))
fig = sns.barplot(x='Depart', y='Venda_Semanal', data=df, hue='Depart', legend = False, palette='Spectral')

#No primeiro gráfico, um departamento entre 60-80(digamos 72), tem valores maiores de vendas.
#Mas nas médias, o departamento 92 tem vendas maiores.
#Departamento 72 pode ser sazonal.
#Tem valores maiores em algumas épocas mas na média o 92 é maior.

# Agora com lojas
x = df['Loja']
y = df['Venda_Semanal']
plt.figure(figsize=(15,5))
plt.title('Venda Semanal por Loja')
plt.xlabel('Lojas')
plt.ylabel('Venda Semanal')
plt.scatter(x,y)
plt.show()

plt.figure(figsize=(20,6))
fig = sns.barplot(x='Loja', y='Venda_Semanal', data=df, hue='Loja', legend = False, palette='Spectral')

# O mesmo ocorre com lojas. No primeiro gráfico, algumas lojas com vendas maiores.
# Mas a maior médio é a loja 20, seguindo a 4 e 14

######### Mudando Data para Datetime e Novas Colunas #########
df["Data"] = pd.to_datetime(df["Data"]) # converte para datetime
df['semana'] =df['Data'].dt.isocalendar().week # dt.isocalendar().week para extrair o número da semana
df['semana'] = df['semana'].astype(str) # converte semana para str
df['mês'] =df['Data'].dt.month
df['ano'] =df['Data'].dt.year

df.groupby('mês')['Venda_Semanal'].mean() # Ver os melhores meses para média de vendas

df.groupby('ano')['Venda_Semanal'].mean() # Ver os melhores anos para média de vendas

vendas_mensais = pd.pivot_table(df, values = "Venda_Semanal", columns = "ano", index = "mês")
vendas_mensais.plot()

# Pelo gráfico, 2011 foi abaixo de 2010 em vendas.
# Na média, 2010 parece ser maior, mas 2012 não tem informaçãoes sobre Novembro e Dezembro que são as maiores vendas.
# Mesmo 2012 sem dois meses, sua média é próxima de 2010.
# Provavelmente, ficaria em primeiro se adicionarmos os resultados perdidos de 2012.

fig = sns.barplot(x='mês', y='Venda_Semanal', hue='mês', data=df, legend = False, palette='Spectral')

# No gráfico acima, as melhores vendas são em Novembro e Dezembro, como esperado.
# Os maiores valores pertencem a Ação de Graças, mas na média, Dezembro tem os maiores valores.

df.groupby('semana')['Venda_Semanal'].mean().sort_values(ascending=False).head()

# Top 5 médias de venda por semana pertencem às 1-2 semanas antes do Natal, Ação de Graças, Black Friday e fim de Maio, quando escolas fecham.

vendas_semanais = pd.pivot_table(df, values = "Venda_Semanal", columns = "ano", index = "semana")
vendas_semanais.plot()

plt.figure(figsize=(20,6))
fig = sns.barplot(x='semana', y='Venda_Semanal', hue='semana', data=df, legend = False, palette='Spectral')

# Pelos gráficos, a 51th e 47th semanas têm médias maiores com efeitos do Natal, Ação de Graças e Black Friday.



######### Efeitos do Preço da Gasolina, IPC, Desemprego e Temperatura #########
# Preço Gasolina
preco_gasolina = pd.pivot_table(df, values = "Venda_Semanal", index= "Preco_gasolina")
preco_gasolina.plot()

# Temperatura
temp = pd.pivot_table(df, values = "Venda_Semanal", index= "Temperatura")
temp.plot()

# IPC
IPC = pd.pivot_table(df, values = "Venda_Semanal", index= "IPC")
IPC.plot()

# Desemprego
desemprego = pd.pivot_table(df, values = "Venda_Semanal", index= "Desemprego")
desemprego.plot()

# Pelos gráficos, não há padrão significativo para IPC, temperatura, desemprego, preço gasolina vs vendas semanais.
# Não há dados para IPC entre 140-180.

df.to_csv('clean_data.csv') # Cria novo data frame para o csv
# Será necessário um CSV limpo para realizar uma análise de Regressão de Floresta Aleatória (Random Forest Regressor)



####### Random Forest Regressor #######
# A métrica para o projeto será pesada por weighted mean absolute error (WMAE)
# Nesta métrica, o erro em semanas de feriado tem 5 vezes mais peso que semanas normais.
# Portanto, deve-se predizer as vendas em feriados precisamente.

pd.options.display.max_columns=100 # visualização de colunas

df = pd.read_csv('clean_data.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
df['Data'] = pd.to_datetime(df['Data']) # Mudando datetime para dividir se precisar
df.head()

# Codificando Dados
# Para processar os dados, mudarei os feriados para boolean (0-1)
# E substituir as lojas A, B, C por 1, 2, 3

df_codificado = df.copy() # Mantém o dataframe original tomando uma cópia dele
grupo_tipo = {'A':1, 'B': 2, 'C': 3}  # muda A,B,C para 1-2-3
df_codificado['Tipo'] = df_codificado['Tipo'].replace(grupo_tipo)
df_codificado['Super_Bowl'] = df_codificado['Super_Bowl'].astype(bool).astype(int) # Muda T,F para 0-1
df_codificado['Ação_de_Graças'] = df_codificado['Ação_de_Graças'].astype(bool).astype(int) # Muda T,F para 0-1
df_codificado['Trabalhador'] = df_codificado['Trabalhador'].astype(bool).astype(int) # Muda T,F para 0-1
df_codificado['Natal'] = df_codificado['Natal'].astype(bool).astype(int) # Muda T,F para 0-1
df_codificado['Feriado'] = df_codificado['Feriado'].astype(bool).astype(int) # Muda T,F para 0-1
df_novo = df_codificado.copy() # Toma cópia do codificado df para manter original


## Observação das interações entre características ##
# Primeiro, retiro colunas de feriados divididas dos meus dados e tento sem.
# Para manter o codificado seguro, designo meu dataframa para o novo e o utilizo.

drop_col = ['Super_Bowl','Trabalhador','Ação_de_Graças','Natal']
df_novo.drop(drop_col, axis=1, inplace=True) # Retiro colunas

plt.figure(figsize = (12,10))
sns.heatmap(df_novo.corr().abs())    # Ver correlações
plt.show()

# Temperatura, desemprego, IPC não têm efeito sobre vendas semanais, então retiro.
# Desconto 4 e 5 muito correlatos com 1. Retiro para evitar multicolinearidade.

drop_col = ['Temperatura','Desconto4','Desconto5','IPC','Desemprego']
df_novo.drop(drop_col, axis=1, inplace=True) # Retira colunas

plt.figure(figsize = (12,10))
sns.heatmap(df_novo.corr().abs())    # Nova correlação sem as colunas anteriores
plt.show()

# Tamanho e tipo são altamente correlatos com vendas semanais.
# Departamento e loja são correlatos com vendas.

df_novo = df_novo.sort_values(by='Data', ascending=True) # Organiza de acordo com data

## Criando divisão Treino-Teste ##
# Coluna data tem valores contínuos, para manter as características de data contínua.
# Não farei divisão aleatória. Divido os dados manualmente por 70%.

treino_dados = df_novo[:int(0.7*(len(df_novo)))] # Pegando a parte treino
teste_dados = df_novo[int(0.7*(len(df_novo))):] # Pegando a parte teste

alvo = "Venda_Semanal"
cols_usadas = [c for c in df_novo.columns.to_list() if c not in [alvo]] # todas colunas exceto venda_semanal

X_treino = treino_dados[cols_usadas]
X_teste = teste_dados[cols_usadas]
y_treino = treino_dados[alvo]
y_teste = teste_dados[alvo]

X = df_novo[cols_usadas] # manter valores X de treino e teste juntos

# Temos informação suficiente em datas como semana do ano.
# Portanto, retiro coluna Data
X_treino = X_treino.drop(['Data'], axis=1) # Retiro data de treino
X_teste = X_teste.drop(['Data'], axis=1) # Retiro data de teste


## Função de definição métrica ##
# Nossa métrica não é calculada como padrão por modelos prontos.
# É erro pesado portanto, uso a função abaixo para calculá-la.

def wmae_test(teste, pred): # WMAE para teste
    pesos = X_teste['Feriado'].apply(lambda feriado:5 if feriado else 1)
    erro = np.sum(pesos * np.abs(teste - pred), axis=0) / np.sum(pesos)
    return erro

## Regressão de Floresta Aleatória ##
# Escolhi os parâmetros manualmente porque com gridsearch iria levar muito tempo.

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()


# Faz pipe tp usar scaler e regressor juntos
pipe = make_pipeline(scaler,rf)

pipe.fit(X_treino, y_treino)

# Predições no set de treino
y_pred = pipe.predict(X_treino)

# Predições no set de teste
y_pred_teste = pipe.predict(X_teste)


wmae_test(y_teste, y_pred_teste)

# Primeira tentativa, erro pesado por volta de 5850.

## Ver a importância das Características
X = X.drop(['Data'], axis=1) # Retira coluna Data de X

importancias = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importancias)[::-1]

# Imprime o ranque de características
print("Ranque de Características:")

for f in range(X.shape[1]):
    print("%d. característica %d (%f)" % (f + 1, indices[f], importancias[indices[f]]))

# Gráfico de importância de características da floresta
plt.figure()
plt.title("Importâncias de Características")
plt.bar(range(X.shape[1]), importancias[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# Após gráfico, retiro de 3-4 características menos importantes e tento modelo.
# Melhor resultado quando retiro coluna mês que é altamente correlata com semana.

X1_treino = X_treino.drop(['mês'], axis=1) # Retirando mês
X1_teste = X_teste.drop(['mês'], axis=1)


## Modelo sem Mês
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X1_treino, y_treino)

# Predições no set de treino
y_pred = pipe.predict(X1_treino)

# Predições no set de teste
y_pred_teste = pipe.predict(X1_teste)


wmae_test(y_teste, y_pred_teste)

# Resultado melhor que anterior

## Modelo todos os Dados
# Garantir que o modelo vai aprender pelas colunas retiradas.
# Aplico o modelo para todos os dados codificados novamente.

# Dividindo treino-teste para todo dataset
treino_dados_cod = df_codificado[:int(0.7*(len(df_codificado)))]
teste_dados_cod = df_codificado[int(0.7*(len(df_codificado))):]

alvo = "Venda_Semanal"
cols_usadas1 = [c for c in df_codificado.columns.to_list() if c not in [alvo]] # Todas colunas exceto preço

X_treino_cod = treino_dados_cod[cols_usadas1]
X_teste_cod = teste_dados_cod[cols_usadas1]
y_treino_cod = treino_dados_cod[alvo]
y_teste_cod = teste_dados_cod[alvo]


X_cod = df_codificado[cols_usadas1] # Para juntas novamente treino e teste
X_cod = X_cod.drop(['Data'], axis=1) # Retiro coluna data para todo X
X_treino_cod = X_treino_cod.drop(['Data'], axis=1) # Retiro data de treino e teste
X_teste_cod= X_teste_cod.drop(['Data'], axis=1)

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X_treino_cod, y_treino_cod)

# Predições no set de treino
y_pred_cod = pipe.predict(X_treino_cod)

# Predições no set de teste
y_pred_teste_cod = pipe.predict(X_teste_cod)


wmae_test(y_teste_cod, y_pred_teste_cod)

# Resultados melhores com todos dados, o modelo aprende com as colunas retiradas

###### Importância de Características para todo DataSet Codificado ######
importancias = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importancias)[::-1]

# Imprime o ranque de características
print("Ranque de Características:")

for f in range(X_cod.shape[1]):
    print("%d. característica %d (%f)" % (f + 1, indices[f], importancias[indices[f]]))

# Gráfico de importância de características da floresta
plt.figure()
plt.title("Importâncias de Características (Todo Codificado)")
plt.bar(range(X_cod.shape[1]), importancias[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_cod.shape[1]), indices)
plt.xlim([-1, X_cod.shape[1]])
plt.show()


# Pelo modelo, retiro algumas colunas e rodo o modelo novamente

df_codificado_novo = df_codificado.copy() # Tomo cópia do dado codificado para manter intacto
df_codificado_novo.drop(drop_col, axis=1, inplace=True)


###### Modelo de Acordo com Importância de Características ######
# Divisão treino-teste
treino_dados_cod_novo = df_codificado_novo[:int(0.7*(len(df_codificado_novo)))]
teste_dados_cod_novo = df_codificado_novo[int(0.7*(len(df_codificado_novo))):]

alvo = "Venda_Semanal"
cols_usadas2 = [c for c in df_codificado_novo.columns.to_list() if c not in [alvo]] # Todas colunas exceto preço

X_treino_cod1 = treino_dados_cod_novo[cols_usadas2]
X_teste_cod1 = teste_dados_cod_novo[cols_usadas2]
y_treino_cod1 = treino_dados_cod_novo[alvo]
Y_teste_cod1 = teste_dados_cod_novo[alvo]

#droping date from train-test
X_treino_cod1 = X_treino_cod1.drop(['Data'], axis=1)
X_teste_cod1= X_teste_cod1.drop(['Data'], axis=1)


rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=40,
                           max_features = 'log2',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X_treino_cod1, y_treino_cod1)

# Predições no set de treino
y_pred_cod = pipe.predict(X_treino_cod1)

# Predições no set de teste
y_pred_teste_cod = pipe.predict(X_teste_cod1)


pipe.score(X_teste_cod1,Y_teste_cod1)


wmae_test(Y_teste_cod1, y_pred_teste_cod)


# Resultados melhores com a seleção de característica do dataset codificado inteiro


###### Modelo com Retirada da Coluna Mês ######
# Mesmo dataset anterior, mesmo modelo sem coluna mês

df_codificado_novo1 = df_codificado.copy()
df_codificado_novo1.drop(drop_col, axis=1, inplace=True)
df_codificado_novo1 = df_codificado_novo1.drop(['Data'], axis=1)
df_codificado_novo1 = df_codificado_novo1.drop(['mês'], axis=1)


# Divisão Treino-teste
treino_dados_cod_novo1 = df_codificado_novo1[:int(0.7*(len(df_codificado_novo1)))]
teste_dados_cod_novo1 = df_codificado_novo1[int(0.7*(len(df_codificado_novo1))):]

alvo = "Venda_Semanal"
cols_usadas3 = [c for c in df_codificado_novo1.columns.to_list() if c not in [alvo]] # Todas colunas exceto preço

X_treino_cod2 = treino_dados_cod_novo1[cols_usadas3]
X_teste_cod2 = teste_dados_cod_novo1[cols_usadas3]
y_treino_cod2 = treino_dados_cod_novo1[alvo]
y_teste_cod2 = teste_dados_cod_novo1[alvo]


# modelagem
pipe = make_pipeline(scaler,rf)

pipe.fit(X_treino_cod2, y_treino_cod2)

# Predições no set de treino
y_pred_cod = pipe.predict(X_treino_cod2)

# Predições no set de teste
y_pred_teste_cod = pipe.predict(X_teste_cod2)


pipe.score(X_teste_cod2,y_teste_cod2)


wmae_test(y_teste_cod2, y_pred_teste_cod)


# Resultados não foram melhores

df_resultados = pd.DataFrame(columns=["Modelo", "Info",'WMAE']) # df resultado para mostrar resultados juntos

# Coloca resultados na df
nova_linha = pd.DataFrame({     
     "Modelo": ['Random Forest Regressor'],
      "Info": ['Sem coluna feriado dividida'], 
       "WMAE" : [5850]})
df_resultados = pd.concat([df_resultados, nova_linha], ignore_index=True)

nova_linha = pd.DataFrame({     
     "Modelo": ['Random Forest Regressor'],
      "Info": ['Sem coluna mês'], 
       "WMAE" : [5494]})
df_resultados = pd.concat([df_resultados, nova_linha], ignore_index=True)

nova_linha = pd.DataFrame({     
     "Modelo": ['Random Forest Regressor'],
      "Info": ['Dados completos'], 
       "WMAE" : [2450]})
df_resultados = pd.concat([df_resultados, nova_linha], ignore_index=True)

nova_linha = pd.DataFrame({     
     "Modelo": ['Random Forest Regressor'],
      "Info": ['Dados completos com seleção de características'], 
       "WMAE" : [1801]})
df_resultados = pd.concat([df_resultados, nova_linha], ignore_index=True)

nova_linha = pd.DataFrame({     
     "Modelo": ['Random Forest Regressor'],
      "Info": ['Dados completos com seleção de características sem mês'], 
       "WMAE" : [2093]})
df_resultados = pd.concat([df_resultados, nova_linha], ignore_index=True)
df_resultados


# Melhores resultados com data set inteiro e seleção de características


###### Modelos de Séries Temporais ######
df.index = df['Data']
df.head()

df["Data"] = pd.to_datetime(df["Data"]) # Mudar data para datarime para decompor

## Gráficos de Vendas ##
plt.figure(figsize=(16,6))
df['Venda_Semanal'].plot()
plt.show()

# Muitos dados repetidos. Deve-se uní-los semanalmente.

df_semana = df['Venda_Semanal'].resample('W').mean()  # 'W' para semanal
plt.figure(figsize=(16, 6))
df_semana.plot()
plt.title('Média de Vendas - Semanal')
plt.show()


# Com coleta de dados semanais, pode-se ver melhor as vendas médias.
# Para ver padrão mensal, reamostrar dados para mensais também.

df_mes = df['Venda_Semanal'].resample('MS').mean()  # 'MS' para mensal (primeiro dia do mês)
plt.figure(figsize=(16, 6))
df_mes.plot()
plt.title('Média de Vendas - Mensal')
plt.show()


# Após mudar dados para mensais, perde-se certos padrões de dados semanais.
# Continuo com dados semanais reamostrados.
df_semana = df_semana.reset_index()		# Transformo index em coluna
df_semana.index = df_semana['Data']
df_semana


## Observar Média Móvel e o Desvio Padrão de 2 Semanas ##
# Busco uma versão mais estacionária, pois os dados são não-estacionários
media_movel = df_semana['Venda_Semanal'].rolling(window=2, center=False).mean()
std_movel = df_semana['Venda_Semanal'].rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_semana['Venda_Semanal'], color='blue',label='Média de Venda Semanal')
ax.plot(media_movel, color='red', label='Média Móvel de 2 Semanas')
ax.plot(std_movel, color='black', label='Desvio-Padrão Móvel de 2 Semanas')
ax.legend()
fig.tight_layout()


###### Teste Adfuller por garantia ######
adfuller(df_semana['Venda_Semanal'])

## Pelo teste os dados são estacionários (p-value < 0.05) ##
# Porém, irei colocar métodos que tornariam os dados mais estacionários, caso precise em outro dataset

###### Divisão Treino-Teste de Dados Semanais ######
# Para dividir continuamente, divido manualmente, não aleatório.

dados_treino = df_semana[:int(0.7*(len(df_semana)))] 
dados_teste = df_semana[int(0.7*(len(df_semana))):]

print('Treino:', dados_treino.shape)
print('Teste:', dados_teste.shape)


alvo = "Venda_Semanal"
cols_usadas = [c for c in df_semana.columns.to_list() if c not in [alvo]] # todas colunas exceto preço

# atribuo valores treino-teste X e y

X_treino = dados_treino[cols_usadas]
X_teste = dados_teste[cols_usadas]
y_treino = dados_treino[alvo]
y_teste = dados_teste[alvo]


dados_treino['Venda_Semanal'].plot(figsize=(20,8), title= 'Venda_Semanal', fontsize=14)
dados_teste['Venda_Semanal'].plot(figsize=(20,8), title= 'Venda_Semanal', fontsize=14)
plt.show()


# Linha azul é treino, amarela é teste


###### Decompondo Dados Semanais par Observar Sazonalidade ######
decomposto = decompose(df_semana['Venda_Semanal'].values, 'additive', m=20) # decompõe dados semanais

decomposed_plot(decomposto, figure_kwargs={'figsize': (16, 10)})
plt.show()

# A cada 20 passos, a sazonalidade converge ao início


###### Tornando Dados Mais Estacionários (APENAS PARA DEMONSTRAÇÃO) ######
# Mostrarei um modelo com dados diferenciados, registrados e deslocados

# Diferenciado
df_semana_dif = df_semana['Venda_Semanal'].diff().dropna() # cria valores de diferença

# Média e Std de dados diferenciados
dif_media_movel = df_semana_dif.rolling(window=2, center=False).mean()
dif_std_movel = df_semana_dif.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_semana_dif, color='blue',label='Diferença')
ax.plot(dif_media_movel, color='red', label='Média Móvel')
ax.plot(dif_std_movel, color='black', label='Desvio-Padrão Móvel')
ax.legend()
fig.tight_layout()


## Deslocado
df_semana_lag = df_semana['Venda_Semanal'].shift().dropna() # desloca valores

# Média e Std de dados deslocados
lag_media_movel = df_semana_lag.rolling(window=2, center=False).mean()
lag_std_movel = df_semana_lag.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_semana_lag, color='blue',label='Diferença')
ax.plot(lag_media_movel, color='red', label='Média Móvel')
ax.plot(lag_std_movel, color='black', label='Desvio-Padrão Móvel')
ax.legend()
fig.tight_layout()


## Registrado
semana_registrada = np.log1p(df_semana['Venda_Semanal']).dropna() # Toma registro dos dados

log_media_movel = semana_registrada.rolling(window=2, center=False).mean()
log_std_movel = semana_registrada.rolling(window=2, center=False).std()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(semana_registrada, color='blue',label='Diferença')
ax.plot(log_media_movel, color='red', label='Média Móvel')
ax.plot(log_std_movel, color='black', label='Desvio-Padrão Móvel')
ax.legend()
fig.tight_layout()



# Mostrei dados sem mudanças, depois deslocado, registrado e versão diferenciado
# Dados diferenciados foram melhores. Esses serão usados para Auto-ARIMA MODEL.



###### MODELO Auto-ARIMA ######
# Divisão Treino-Teste
dif_dados_treino = df_semana_dif [:int(0.7*(len(df_semana_dif)))]
dif_dados_teste = df_semana_dif [int(0.7*(len(df_semana_dif))):]

model_auto_arima = auto_arima(dif_dados_treino, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=20, max_q=20, max_P=20, max_Q=20, seasonal=True,maxiter=200,
                  information_criterion='aic',stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = False)
model_auto_arima.fit(dif_dados_treino)

y_pred = model_auto_arima.predict(n_periods=len(dif_dados_teste))
y_pred = pd.DataFrame(y_pred,index = dados_teste.index,columns=['Prediction'])
plt.figure(figsize=(20,6))
plt.title('Predição de Vendas Semanais Usando Auto-ARIMA', fontsize=20)
plt.plot(dif_dados_treino, label='Treino')
plt.plot(dif_dados_teste, label='Teste')
plt.plot(y_pred, label='Predição do ARIMA')
plt.legend(loc='best')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Venda Semanal', fontsize=14)
plt.show()

# Caso tente outro Modelo

###### MODELO ExponentialSmoothing ######
# Verifico modelos Holt-Winters de acordo com dados.
# Exponential Smooting é usado quando os dados tem tendência, e achata a tendência.
# Método de tendência amortecida adiciona um parâmetro de amortecimento para que a tendência convirja para um valor constante no futuro.

# Os dados têm certos valores negativos e zero, uso sazonal aditivo e tendência em vez do multiplicativo.
# Períodos sazonais escolhidos a partir dos gráficos compostos acima.
# Ajustar modelo com iterações toma muito tempo, mudo e experimento modelo
# para diferentes parâmetros, encontro os melhores e ajusto modelo.

model_holt_winters = ExponentialSmoothing(dif_dados_treino, seasonal_periods=20, seasonal='additive',
                                           trend='additive',damped=True).fit() #Tendência e sazonalidade aditiva.
y_pred = model_holt_winters.forecast(len(dif_dados_teste))# Prediz valores de teste

# Visualiza treino, teste e dados preditos.
plt.figure(figsize=(20,6))
plt.title('Predição de Vendas Semanais Usando ExponentialSmoothing', fontsize=20)
plt.plot(dif_dados_treino, label='Treino')
plt.plot(dif_dados_teste, label='Teste')
plt.plot(y_pred, label='Predição do ExponentialSmoothing')
plt.legend(loc='best')
plt.xlabel('Data', fontsize=14)
plt.ylabel('Venda Semanal', fontsize=14)
plt.show()
