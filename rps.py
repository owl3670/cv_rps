import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def train_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
        rescale=1. / 255,
        rotation_range=45,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.4
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.4)

    train_generator = training_datagen.flow_from_directory(
        # YOUR CODE HERE
        TRAINING_DIR,
        batch_size=32,
        target_size=(150, 150),
        class_mode='categorical',
        subset='training',
    )

    validation_generator = test_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=32,
        target_size=(150, 150),
        class_mode='categorical',
        subset='validation',
    )

    model = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "tmp_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_generator,
              steps_per_epoch=32,
              epochs=30,
              validation_data=(validation_generator),
              validation_steps=8,
              callbacks=[checkpoint]
              )

    model.load_weights(checkpoint_path)
    return model


# Extract hand features and Classify rps from the live webcam using tensorflow model
def classify_rps(model_path):
    model = load_model(model_path)

    moves_dict = {0: 'paper', 1: 'rock', 2: 'scissors'}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)
        roi = frame[100:400, 100:400]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)
        pred = model.predict(img)
        move_code = np.argmax(pred[0])
        rps_move = moves_dict[move_code]
        cv2.putText(frame, "Your Move: " + rps_move, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors", frame)
        k = cv2.waitKey(10)

        # esc key 입력시 종료
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    model_path = 'tmp/rps.h5'
    # model = train_model()
    # model.save(model_path)
    classify_rps(model_path)


if __name__ == '__main__':
    main()
