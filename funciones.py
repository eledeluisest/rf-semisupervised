"""
Random Forest no supervisado. Cálculo de la matriz de proximidad.
05/06/2018

"""
import pandas as pd
import numpy as np
def shuffle(df_, seed=1234):
    """
    Esta función resamplea de forma independiente cada una de las columnas del dataframe que le pases y las concatena
    para generar otro DataFrame del mismo tamaño pero con los valores generados aleatoriamente.
    :param df_: Dataframe de pandas
    :param seed: Semilla para el resampleo
    :return: DataFrame resampleado
    """
    import pandas as pd
    df = pd.DataFrame()
    #Esta i va a servir para cambiar la semilla de forma que las diferentes columnas no se resampleen siempre igual
    i=10
    #Vamos por cada columna desordenandola y concatenandola a la anterior
    for key in df_.keys():
        df = pd.concat([df,df_[key].sample(frac=1,random_state=seed+i).reset_index(drop=True)],axis=1)
        i+=10
    return df

def genera_muestra_sintetica(X,generator='resampling',seed_=1234):
    """
    Esta función sirve para generar la muestra sintética del RF no supervisado.
    :param X: Muestra de train original
    :param generator: Ahora mismo solo está implementado el resampleo. Otros métodos son:
     -Aprender la distribución de los valores de cada columna y generar valores sigueindo dichas distribución
     -Elegir puntos aleatorios de una caja n-dimensional definida por los valores máximos de cada columna
     todo: Estos métodos habría que implementarlos porque, según la literatura, son muy determinantes en la matriz de proximidad
     final
    :param seed_: Semilla para la función shuffle
    :return: Devuelve la muestra sintética y su target.
    """
    import numpy as np
    import pandas as pd
    if generator == 'resampling':
        X_ = X.copy()
        X_ = shuffle(X_,seed=seed_)
        y_sintetico = pd.Series(np.array([0]*len(X)).reshape(-1,))

    return (X_,y_sintetico)

def prepara(df_):
    """
    Esta función va por cada columna del dataset y saca dummies de las que sean cadena de caracteres.
    todo: Habría que implementar todas las funcionalidades necesarias para que funcionara el algoritmo en cualquier situación
    (relleno de missings, estandarización si se desea, reducción de dimensiones, especificación de variables categoricas
    ordinales...)
    :param df_: DataFrame a preparar
    :return: DataFrame tratado
    """
    df = df_.copy()
    for key in df.keys():
        if df[key].dtype == object:
            df = pd.concat([df,pd.get_dummies(df[key])],axis=1)
            del df[key]
    return df

def prox_matrix_unsupervised(X,target, trees = 10):
    """
    Esta función ejecuta el Random Forest
    :param X: Muestra original
    :param trees: Número de árboles para el clasificador
    :return: Matriz de proximidad y matriz generada en el conteo
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd

    # Generamos la muestra sintética utilziando la función antetior
    X_sint, y_sint = genera_muestra_sintetica(X)

    # Creamos el target para la muestra original y generamos los dataset para el random forest no supervisado (concatenar
    # muestra original y sintética)
    y_original = pd.Series(np.array([1]*len(X)).reshape(-1,))
    X_unsupervised = pd.concat([X,X_sint],axis=0)
    y_unsupervised = pd.concat([y_original,y_sint],axis=0)

    # Le pasamos el procesado de datos
    X_unsupervised = prepara(X_unsupervised)

    # Definimos y entrenamos el clasificador
    clf = RandomForestClassifier(n_estimators=trees, n_jobs=-1,oob_score=True,random_state=1234)
    clf.fit(X_unsupervised, y_unsupervised)

    """
    todo: Este método solo se pude aplicar siempre y cuando el clasificador sea capaz de distinguir correctamente entre la 
    muestra real y la sintética así que en el futuro debería haber un filtro que comprobara si realmente el clasificador
    lo ha hecho bien
    print(pd.concat([pd.Series(X_unsupervised.keys()).reset_index(drop=True),
                        pd.Series(clf.feature_importances_).reset_index(drop=True).rename(
                            'importancias')],axis=1).sort_values(by='importancias',ascending=False).head())
    print('AUC SCORE DE LA CLASIFICACION NO SUPERVISADA:',roc_auc_score(y_unsupervised,clf.predict_proba(X_unsupervised)[:,1]))
    """
    # Cluster dimensión define los registros de los cuales nos interesa calcular la proximidad
    cluster_dimension = len(X)

    # Em esta lista vamos a almacenar las posiciones en las que el target es igual a uno (me daba error meterlo en el .loc)
    # Hacemos esto para calcular solamente la distancia a los unos y no a todos los elementos
    unos_pos = []
    for x in target:
        if x == 1:
            unos_pos.append(True)
        else:
            unos_pos.append(False)

    # Sacamos una matriz UNOS de dimension U*T donde U es el número de unos en el target y T es el número de árboles en el bosque
    leaves_unos = pd.DataFrame(clf.apply(X_unsupervised)).loc[unos_pos, :]
    # Sacamos una matriz TODOS de dimension N*T donde N es el número de elementos en la muestra y T el número de árboles
    leaves = clf.apply(X_unsupervised)[:cluster_dimension, :]
    # El método apply nos devuelve el último nodo en el que cayó cada registro. Así será fácil calcular la relación entre
    # registros (+1 si dos registros caen en el mismo nodo) Debemos ejecutarla sobre el conjunto de entrenamiento al completo

    proba = []
    # Ahora, para cada elemento de UNOS vamos a ver las coincidencias en TODOS. Esto nos da una matriz de dimensión N*T con valor
    # true/false en función si para cada elemento y arbol coincidió el nodo final entre el registro de UNOS y el REGISTRO de todos.
    # Despueés sumamos todas las filas para obtener un array de longitud N que nos da la distancia de cada elemento a un uno concreto
    # e iteramos al siguiente uno. Al final tendremos una lista con U elementos con un array de distancias en cada elemento.
    for uno in leaves_unos.iterrows():
        proba.append(np.sum(np.logical_not(leaves == uno[1].values), axis=1))

    #value in cluster dimension (only original sample)

    return pd.DataFrame(proba)
