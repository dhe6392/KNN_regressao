import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv(r'C:\Users\andre\OneDrive\Área de Trabalho\Python\IA\cap14_Kn\train_data.csv')
df.dropna(inplace=True)

#-----------------separa features da variavel alvo e normaliza aquelas----------------------------
X_cru = df.drop(['country_name','waste'],axis=1)
scaler = StandardScaler()
scaler.fit(X_cru)
scaled_features = scaler.transform(X_cru)

#-----------------mastiga os dados para poderem ser treinados----------------------------------------
X = pd.DataFrame(data=scaled_features,columns=X_cru.columns)
Y = df['waste']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

#----------testa todos os n_neighbors possiveis para achar o que minimiza a taxa de erro------------
ajuste = [None]*len(X_train)
pred = [None]*len(X_train)
K = [i for i in range(1,len(X_train)+1)]
for k in K:
    knn = KNeighborsRegressor(n_neighbors = k,weights='distance')#dou pesos maiores para vizinhos mais proximos
    knn.fit(X_train,Y_train)
    pred[k-1] = knn.predict(X_test)
    ajuste[k-1] = knn.score(X_test,Y_test) #mede R^2: qual % da variacao dos dados o modelo consegue explicar

k_otimizado = np.argmax(ajuste)+ 1
R_quadrado = ajuste[k_otimizado-1]
print(f'K: {k_otimizado} vizinhos\nAjuste (R^2): {R_quadrado*100:.1f}%')

#---------Mostra graficamente que um baixo K gera overfitting e um alto gera underfitting, havendo assim um meio termo ideal
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.set_xlabel('n_neighbors')
axes.set_ylabel('Ajuste (R^2)')
axes.set_title('Precisao do KNN')
axes.plot(K, ajuste, 'b')
axes.axvline(x=k_otimizado, color='red', linestyle='--', label=f'k ótimo = {k_otimizado}')
axes.legend()
plt.savefig('cap14_Kn/Paises.png')
plt.show()