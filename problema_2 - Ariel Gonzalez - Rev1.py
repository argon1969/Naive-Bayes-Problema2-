import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn

def plot_boundaries(X_train, X_test, y_train, y_test, score, probability_func, h = .02, ax = None):
    X = np.vstack((X_test, X_train))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    Z = probability_func(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, 50, cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6, s=200)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')

    
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

''' 
Utilizando las siguientes funciones/paquetes resolver los siguientes problemas de clasificación en **problema_2.py**:

**genfromtxt** de **numpy** para leer los dos datasets:
- ./datasets/student_admission.txt
- ./datasets/chip_tests.txt

**train_test_split** de **sklearn** para dividir entre test set y train set. Recomendamos un 40% de datos para test set

**GaussianNB** de **sklearn** como modelo de ML.

Y la función **plot_boundaries(X_train, X_test, y_train, y_test, score, predict_proba, ax=ax, h=h)** incluida en problema_2.py para graficar los resultados. X_train es un np.array con los features de entrada, y_train es la etiqueta. Lo mismo con X_test e y_test, Score es el 'accuracy' del modelo, predict_proba es la función que dada una entrada de la probabilidad de clasificar correcto y h es el paso para la grafica del 'boundary' 
'''

def train_and_plot(X, y, h=1):
    # TODO

    X_train, X_test, y_train, y_test = train_test_split(features_matrix,labels, test_size=0.4, random_state=42)

    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import make_moons, make_circles, make_classification
    import matplotlib.pyplot as plt

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    linearly_separable = (X_test, y_test)

    datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

    position=0

    ax = plt.subplot(len(datasets), 1, position+1)
    plot_boundaries(X_train, X_test, y_train, y_test, score, clf.predict_proba, ax=ax)

    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(features_matrix,labels, test_size=0.4, random_state=42)

    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import make_moons, make_circles, make_classification
    import matplotlib.pyplot as plt

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    linearly_separable = (X_test, y_test)

    datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

    position=0

    ax = plt.subplot(len(datasets), 1, position+1)
    plot_boundaries(X_train, X_test, y_train, y_test, score, clf.predict_proba, ax=ax)

    plt.show()

    return
 
# Primer dataset -------------------------------------------------------------------------------------------

data_student=np.genfromtxt("./datasets/student_admission.txt",names = ['A','B','categoria'], delimiter=",")
print (data_student)

features_matrix = np.zeros((len(data_student),2))
labels = np.empty((len(data_student)), dtype=bool)

for i, person in enumerate(data_student):
    features_matrix[i, 0] = person['A']
    features_matrix[i, 1] = person['B']
        
    if person['categoria']==1:
            labels[i] = True
    else:
            labels[i] = False
                
            
print('Matriz y lista con los datos:')
print(features_matrix)
print(labels)

train_and_plot(features_matrix,labels)

# Segundo dataset --------------------------------------------------------------------------------------
   
data_student=np.genfromtxt("./datasets/chip_tests.txt",names = ['A','B','categoria'], delimiter=",")
print (data_student)

features_matrix = np.zeros((len(data_student),2))
labels = np.empty((len(data_student)), dtype=bool)

for i, person in enumerate(data_student):
    features_matrix[i, 0] = person['A']
    features_matrix[i, 1] = person['B']
        
    if person['categoria']==1:
            labels[i] = True
    else:
            labels[i] = False
                
            
print('Matriz y lista con los datos:')
print(features_matrix)
print(labels)

train_and_plot(features_matrix,labels)


X_train, X_test, y_train, y_test = train_test_split(features_matrix,labels, test_size=0.4, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt

clf = GaussianNB()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

linearly_separable = (X_test, y_test)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

position=0

ax = plt.subplot(len(datasets), 1, position+1)
plot_boundaries(X_train, X_test, y_train, y_test, score, clf.predict_proba, ax=ax)

plt.show()











