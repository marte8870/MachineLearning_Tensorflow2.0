Análisis de Datos y Aprendizaje Máquina con Tensorflow 2
==========================


#### *Nota1: Mantenimiento completo
#### *Nota2: Para última actualización, descargar siempre el commit más reciente de este repositorio



# Instalación
## Ambiente local

Primero instalar [Anaconda](https://www.anaconda.com/) y [git](https://git-scm.com/). Después clonar este repositorio escribiendo en la terminal los siguientes comandos:

    $ cd $HOME  # o el directorio de preferencia
    $ git clone https://github.com/marte8870/MachineLearning_Tensorflow2.0.git


### Python y Bibliotecas Requeridas

Usando Anaconda

- Crear ambiente de trabajo


    $ conda create -n myenv tensorflow scipy matplotlib pandas scikit-learn seaborn pydot pydotplus pillow graphviz
    $ source activate myenv 
    
    
- Para gpu usar tensorflow-gpu

## Colab

- [Google Colab](https://colab.research.google.com/)

###  Jupyter
    $ conda install -c anaconda jupyter 
    $ jupyter notebook

## Conjuntos de datos que se analizan
- [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- [MNIST handwritten digit](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Reuters newswire classification](https://keras.io/datasets/#reuters-newswire-topics-classification)
- [IMDB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


## Obtener otros datasets
- [Kaggle](https://www.kaggle.com/)
- [UCI](http://archive.ics.uci.edu/ml/index.php)




# Contenido

### Pre-procesamiento y manejo de datos para Aprendizaje Máquina



- [Python, Numpy y Pandas en ejecución](./1.Pre-procesamiento/PdNumpy.ipynb)
- [Algebra Lineal](./1.Pre-procesamiento/Álgebra-Lineal.ipynb)
- [Estadística](./1.Pre-procesamiento/Estadística.ipynb)
- [Exploración de Datos](./1.Pre-procesamiento/Pandas.ipynb)
- [Visualización de Datos](./1.Pre-procesamiento/Visualización.ipynb)


### Aprendizaje Máquina 
#### Clasificación

- [Naive Bayes](./2.Clasificación/Naive-Bayes.ipynb)
- [Decision Trees](./2.Clasificación/ID3.ipynb)
- [KNN (parameter tuning)](./2.Clasificación/KNN.ipynb)
- [Evaluación/Regresión Logística *](./2.Clasificación/Evaluación-Regresion-Logistica.ipynb)

#### Reducción de Dimensionalidad y Clustering

- [PCA](./3.Clustering-ReducciónDimensionalidad/PCA.ipynb)
- [K-Means](./3.Clustering-ReducciónDimensionalidad/K-means.ipynb)


### Deep Learning 
#### Perceptrón Multicapa

- [Estructura MLP](./4.PerceptrónMulticapa-Regularización/Estructura-MLP.ipynb)
- [MLP Imperative/Training loop](./4.PerceptrónMulticapa-Regularización/Estructura-MLP-OOP.ipynb)
- [Función costo/activación - Inicialización](./4.PerceptrónMulticapa-Regularización/Costo-Activación.ipynb)


#### Regularización y Optimización 

- [Optimizadores](./4.PerceptrónMulticapa-Regularización/Optimizadores.ipynb)
- [Batch Normalization](./4.PerceptrónMulticapa-Regularización/Batch-Norm.ipynb)
- [Dropout](./4.PerceptrónMulticapa-Regularización/Dropout.ipynb)
- [L2](./4.PerceptrónMulticapa-Regularización/L2.ipynb)
- [Evaluación/MLP *](./4.PerceptrónMulticapa-Regularización/Evaluación-MLP.ipynb)


#### Redes Neuronales Recurrentes (Procesamiento de secuencias)
- [Estructura RNN/LSTM](./5.ProcesamientoSecuencias/RNN.ipynb)
- [RNN Imperative/Training loop](./5.ProcesamientoSecuencias/RNN-OOP.ipynb)
- [Clasificación de Texto](./5.ProcesamientoSecuencias/Clasificar-Texto.ipynb)
- [Deep-Bidirectional RNN/Regularización](./5.ProcesamientoSecuencias/Deep-Bidirectional-RNN.ipynb)


#### Redes Neuronales Convolucionales (Procesamiento de imágenes)
- [Estructura CNN](./6.ProcesamientoImágenes/CNN.ipynb)
- [CNN Imperative/Training loop](./6.ProcesamientoImágenes/CNN-OOP.ipynb)
- [Deep CNN/Regularización](./6.ProcesamientoImágenes/CNN2.ipynb)
- [Computer Vision (Clasificación de objetos)](./6.ProcesamientoImágenes/Computer-Vision.ipynb)
- [VGG19-NASNetLarge (Modelos pre-entrenados)](./6.ProcesamientoImágenes/VGG19-NASNetLarge.ipynb)
- [Autoencoders](./6.ProcesamientoImágenes/Autoencoder.ipynb)




[Solución MLP](./4.PerceptrónMulticapa-Regularización/Solución-MLP.ipynb) y [solución Regresión Logística](./2.Clasificación/Solución-Regresion-Logistica.ipynb)
