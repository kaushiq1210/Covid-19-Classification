from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setText("Browse Image")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setText("            COVID-19 DETECTION")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setText("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setText("Recognized Class")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setText("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("COVID-19 Detection")

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)

    def classifyFunction(self):
        try:
            with open("model.json", "r") as json_file:
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.weights.h5")

            with open("class_indices.json", "r") as f:
                class_indices = json.load(f)
            index_to_class = {v: k for k, v in class_indices.items()}

            img_path = self.file
            test_image = image.load_img(img_path, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image /= 255.0

            result = loaded_model.predict(test_image)
            predicted_index = int(np.argmax(result))
            confidence = float(np.max(result))
            predicted_label = index_to_class[predicted_index]

            print("Raw prediction vector:", result)

            if confidence < 0.60:
                self.textEdit.setText(f"Uncertain prediction: {predicted_label} ({confidence*100:.2f}%)")
            else:
                self.textEdit.setText(f"{predicted_label} ({confidence*100:.2f}%)")

        except Exception as e:
            self.textEdit.setText("Classification failed.")
            print("Error during classification:", e)

    def trainingFunction(self):
        try:
            self.textEdit.setText("Training in progress...")

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Conv2D(96, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(2, activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1./255)

            training_set = train_datagen.flow_from_directory(
                '/Users/sivaramans/College/Novitech AI Projects/Covid - 19  Code/TrainingDataset',
                target_size=(128, 128),
                batch_size=8,
                class_mode='categorical'
            )

            test_set = test_datagen.flow_from_directory(
                '/Users/sivaramans/College/Novitech AI Projects/Covid - 19  Code/TestingDataset',
                target_size=(128, 128),
                batch_size=8,
                class_mode='categorical',
                shuffle=False
            )

            # Calculate class weights
            labels = training_set.classes
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
            class_weights = dict(enumerate(class_weights))

            # Save class index mapping
            with open("class_indices.json", "w") as f:
                json.dump(training_set.class_indices, f)

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(
                training_set,
                steps_per_epoch=100,
                epochs=50,
                validation_data=test_set,
                validation_steps=50,
                callbacks=[early_stop],
                class_weight=class_weights
            )

            # Evaluate model
            Y_pred = model.predict(test_set)
            y_pred = np.argmax(Y_pred, axis=1)
            print('Confusion Matrix')
            print(confusion_matrix(test_set.classes, y_pred))
            print('Classification Report')
            target_names = list(training_set.class_indices.keys())
            print(classification_report(test_set.classes, y_pred, target_names=target_names))

            # Save model
            with open("model.json", "w") as json_file:
                json_file.write(model.to_json())
            model.save_weights("model.weights.h5")
            self.textEdit.setText("Training completed and model saved.")

        except Exception as e:
            self.textEdit.setText("Training failed.")
            print("Error during training:", e)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
