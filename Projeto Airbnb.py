import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib
pd.set_option('display.max_columns', None)

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = Path(r'C:/Users/natha\Projeto 3 - Ciência de Dados - Aplicação de Mercado de Trabalho/dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nomes_mes = arquivo.name[:3]
    mes = meses[nomes_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

    df = pd.read_csv(caminho_bases / arquivo.name, low_memory= False)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb._append(df)

#como temos muitas colunas, nosso modelo pode acabar ficando muito lento.
#Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão, por isso,
#vamos excluir algumas colunas da nossa base

#tipos de colunas que vamos excluir:
    # 1.IDs, Links e informações não relevantes para o modelo
    # 2.Colunas repetidas pu extremamente parecida com outra(que dão a mesma informação para o modelo. Ex: Data x Ano/Mês
    # 3.Colunas preenchidas com texto livre - Não rodaremos nenhuma análise de palavras ou algo do tipo
    # 4.Colunas em que todos os quase todos os valores são iguais

#Para isso, vamos criar um arquivo em excel com os 1.0000 primeiros regstros e fazer uma análise qualitativa

base_airbnb.head(1000).to_csv('primeiros registros.csv', sep = ';')

#Depois da análise quantitativa das colunas, levando em conta os critérios explicados acima, ficamos com as seguintes colunas:

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']
base_airbnb = base_airbnb.loc[:, colunas]


## tratar Valores Faltando
    # Visualizando os dados, percebemos que existe uma grande disparidade em dados faltantes. As colunas com mais de 300.000 valores
    #Nan foram excluidas da análise.
    #Para as outras colunas, como temos muitos dados(mais de 900.00 linhas) vamos excluir as linhas que contém dados Nan
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis = 1)

base_airbnb = base_airbnb.dropna()

#Verificar os Tipos de Dados de cada coluna

    #Como preço e extra people estão sendo reconhecidos como objeto(ao invés de float) temos
    #que mudar o tipo de variável da coluna.
#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy = False)

##extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy = False)


base_airbnb['host_is_superhost'] = base_airbnb['host_is_superhost'].replace({'t': 1, 'f': 0})
base_airbnb['instant_bookable'] = base_airbnb['instant_bookable'].replace({'t': 1, 'f': 0})
base_airbnb['is_business_travel_ready'] = base_airbnb['is_business_travel_ready'].replace({'t': 1, 'f': 0})

print(base_airbnb.dtypes)

##Análise Exploratória e Tratar Outliers

##ver correlação entre as features e decidir quais features mantenho
#plt.figure(figsize= (15, 10))
#sns.heatmap(base_airbnb.corr(), annot= True, cmap='Greens')
#plt.show()

##Definição de funções para análise de Outliers

def limites (coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return [q1 - 1.5 * amplitude, q3 + 1.5 *amplitude]

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_superior = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_superior),:]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

def diagrama_caixa(coluna):
    fig,(ax1, ax2) = plt.subplots(1, 2)  ##para dois gráficos
    fig.set_size_inches(15, 5)             ##para dois gráficos
    sns.boxplot(x = coluna, ax= ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x = coluna, ax = ax2)
    plt.show()

def histograma(coluna):
    plt.figure(figsize= (15,5)) #-- Para um gráfico
    sns.displot(coluna)
    plt.show()

def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))  # -- Para um gráfico
    ax = sns.barplot(x =coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
    plt.show()

def grafico_barra_count(coluna, base):
    plt.figure(figsize= (15, 5))
    grafico = sns.countplot(x=coluna, data= base)                 #gráfico que faz a contagem automaticamente, equivale ao value.counts para df
    grafico.tick_params(axis= 'x', rotation = 90)
    plt.show()

##Price
#diagrama_caixa(base_airbnb['price'])
#histograma(base_airbnb['price'])

    ##como estamos contruindo um modelo para imóveis comuns, acredito que os valores acima do limite superior serão apenas de apartamentos
    ## de luxo, que não é o nosso objetivo principal. Por isso, podemos excluir esses outilers.

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print(linhas_removidas)
histograma(base_airbnb['price'])

##extra people
diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
histograma(base_airbnb['extra_people'])
print(linhas_removidas)

##host_listings_count
diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])

    ##podemos exclui os outilers, pois o objetivo do projeto não é para hosts com mais de 6 imóveis.
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
grafico_barra(base_airbnb['host_listings_count'])
print(linhas_removidas)

#accommodates
diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accomodates'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
grafico_barra(base_airbnb['accommodates'])
print(linhas_removidas)

#bathrooms
diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts()) #tiando os limites do eixo x
plt.show()

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
grafico_barra(base_airbnb['bathrooms'])
print(linhas_removidas)


##beds
diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
grafico_barra(base_airbnb['beds'])
print(linhas_removidas)

##bedrooms
diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedroomss'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
grafico_barra(base_airbnb['bedrooms'])
print(linhas_removidas)

##guests_included
diagrama_caixa(base_airbnb['guests_included'])
grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
    ##vou remover essa feature da análise. Parece que os usuários do airbenb usam
    #muito o valor padrão do airbnb como 1 guest inclued. Isso pod elevar o modelo
    #a considerar uma feature que na verdade não é essencial para definição do preço
    ##por isso me parece melhor excluir a coluna da análise

base_airbnb= base_airbnb.drop('guests_included', axis = 1)  ##excluindo coluna
print(base_airbnb.shape)

##minimum_nights
diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
grafico_barra(base_airbnb['minimum_nights'])
print(linhas_removidas)

#maximum_night
diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])
base_airbnb= base_airbnb.drop('maximum_nights', axis = 1)  ##excluindo coluna
print(base_airbnb.shape)

##number_of_reviews
diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])
base_airbnb= base_airbnb.drop('number_of_reviews', axis = 1)  ##excluindo coluna
print(base_airbnb.shape)

##Tratamento de colunas de valores de texto

    ##PROPERTY_TYPE
print(base_airbnb['property_type'].value_counts())  ##contando por categoria

tabela_tipos_de_casa = base_airbnb['property_type'].value_counts()
coluna_agrupar = []

for tipo in tabela_tipos_de_casa.index:   ##colocando em lista auxiliar
    if tabela_tipos_de_casa[tipo] < 2000:
        coluna_agrupar.append(tipo)

for tipo in coluna_agrupar:                 ##modificando valores na base original
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type' ] = 'Outros'

print(base_airbnb['property_type'].value_counts())
grafico_barra_count('property_type', base_airbnb)

    ##room_type

#Não foi necessário mexer, possui apenas 4 tipos de quartos

    #bed_type

print(base_airbnb['bed_type'].value_counts())  ##contando por categoria

tabela_bed_types = base_airbnb['bed_type'].value_counts()
coluna_agrupar_bed_types = []

for tipo in tabela_bed_types.index:   ##colocando em lista auxiliar
    if tabela_bed_types[tipo] <10000:
        coluna_agrupar_bed_types.append(tipo)

for tipo in coluna_agrupar_bed_types:                 ##modificando valores na base original
    base_airbnb.loc[base_airbnb['bed_type'] == tipo, 'bed_type' ] = 'Outros'

print(base_airbnb['bed_type'].value_counts())
grafico_barra_count('bed_type', base_airbnb)

    ##cancellation_policy

    #agrupando categorias de cancellation_pollicy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
coluna_agrupar_cancellation = []

for tipo in tabela_cancellation.index:   ##colocando em lista auxiliar
    if tabela_cancellation[tipo] < 10000:
        coluna_agrupar_cancellation.append(tipo)
for tipo in coluna_agrupar_cancellation:                 ##modificando valores na base original
    base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy' ] = 'strict'

tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
grafico_barra_count('cancellation_policy', base_airbnb)

    ##amenities
##como temos uma diversidade muito grande de amenitties e, às vezes, as mesmas podem ser escritas de forma diferente,
#vamos avaliar a quantidade como paramentro para o modelo

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb= base_airbnb.drop('amenities', axis = 1)  ##excluindo coluna
print(base_airbnb.shape)


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
grafico_barra(base_airbnb['n_amenities'])
print(linhas_removidas)

##Visualização de Mapa das Propriedades

amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()

##Encoding

##Presisamos ajustar as features para facilitar o trabalho do modelo futuro

#Features de Valores True ou False, vamos substituir True por 1 e False por 0.
#Features de Categoria (features em que os valores da coluna são textos) vamos utilizar o método de encoding de variáveis dummies

colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']

colunas_categorias = []
base_airbnb_cod = base_airbnb.copy()

for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data = base_airbnb_cod, columns= colunas_categorias)

##Análise Exploratória e Tratar Outliers

##ver correlação entre as features e decidir quais features mantenho
plt.figure(figsize= (15, 10))
sns.heatmap(base_airbnb_cod.corr(), annot= True, cmap='Greens')
plt.show()

##modelo de previsão

    #métricas de avaliação

def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    rsme = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR2:{r2:.2%}\nRSME:{rsme:.2f}'

    #escolha dos modelos a serem testados

modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis= 1)

    #seperar os dados em treino e teste + Treino do Modelo

x_train, x_Test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo  in modelos.items():
    #treinar
    modelo.fit(x_train, y_train)
    ##testar
    previsao = modelo.predict(x_Test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


## - Modelo escolhido como melhor modelo: ExtraTrssRegressor
 ##Esse foi o modelo com maior de r2 e ao mesmo tempo o menor
 #valor de RSME. Como não tivemos uma grande diferença de
 ##velocidade de treino e de previsão desse modelo com o modelo
 ## de RandomForest(que teve resultados próximos de r2 e RSME vou escolher o modelo
 #ExtraTress.

 ##O modelo de regressão linear não obteve um resultado satisfatório, com valores de
 #R2 e RSME muito piores do que os modelos.

 #Resultado  das métricas  de avaliação do vencedor

#Modelo
#ExtraTrees:
#R²:97.49 %
#RSME: 41.99

#ajustes e melhorias no modelo
print(modelo_et.feature_importances_)
print(x_train.columns)

importancia_features = pd.DataFrame(modelo_et.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by= 0 , ascending= False)   ##ordenando coluna
print(importancia_features)
plt.figure(figsize= (15, 5))
ax = sns.barplot(x = importancia_features.index, y = importancia_features[0])
ax.tick_params(axis= 'x', rotation = 90)

## Ajustes Finais no Modelo

##is_business_travel não parece ter muito impacto no nosso modelo. Por isso, para chegar em um modelo mais simples, vou excluir
# esta feature e testar o modelo sem ela

base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)
modelo_et.fit(x_train, y_train)
previsao = modelo_et.predict(x_Test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))

#Modelo ExtraTrees:
#R²:97.51%
#RSME:41.84

base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
x = base_teste.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(x_train, y_train)
previsao = modelo_et.predict(x_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


##Deploy do Projeto
x['price'] = y
x.to_csv('dados.csv')

joblib.dump(modelo_et, 'modelo.joblib')

















































































































































































#abril2018_df = pd.read_csv(r'C:\Users\natha\Projeto 3 - Ciência de Dados - Aplicação de Mercado de Trabalho\dataset\dezembro2018.csv')
#print(abril2018_df)