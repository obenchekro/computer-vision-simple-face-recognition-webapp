import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import urllib
import tarfile


def download_dataset(path, filename='wiki.tar'):
    url = "data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename) as tar:
        tar.extractall()
        
class CNNModel:
    def __init__(self, data_path):
        self_path = data_path
        (x_train, y_train), (x_test, y_test) = tf.load_data(self.data_path) # change the path to where you have downloaded the dataset
        self.preprocess_data()
        self.create_model()
        self.compile_model()
        self.train_model()
        
    def preprocess_data(self):
        # Normaliser les données
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Convertir les étiquettes de sexe en catégories binaires (homme / femme)
        self.y_train = np.where(self.y_train == 0, 0, 1)
        self.y_test = np.where(self.y_test == 0, 0, 1)
        
    def create_model(self):
        # Créer un modèle CNN
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        
    def compile_model(self):
        # Compiler le modèle
        self.model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        
    def train_model(self):
        # Entraîner le modèle
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=64)
    
    def evaluate_model(self):
        # Evaluer le modèle
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        return test_loss, test_acc
        
    def predict(self, filepath):
        # Read and preprocess the image
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(32, 32))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0

        # Use the model to make a prediction
        prediction = self.model.predict(np.expand_dims(img, axis=0))[0][0]
        
        return prediction

cnn_model = CNNModel(r"D:\Othmane\Documents\model\dataset\wiki_crop")

prediction = cnn_model.predict(r'D:\Games\Hitman 3\ent\istockphoto-898619084-612x612.jpg')
print(prediction)
