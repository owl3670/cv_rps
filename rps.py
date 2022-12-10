import cv2
import numpy as np
import tensorflow as tf


def classify_rps(model_path):
    model = tf.keras.models.load_model(model_path)

    moves_dict = {0: 'rock', 1: 'paper', 2: 'scissors'}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, (200, 200), (800, 800), (255, 255, 255), 2)
        roi = frame[200:800, 200:800]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)
        pred = model.predict(img)
        move_code = np.argmax(pred[0])
        rps_move = moves_dict[move_code]
        cv2.putText(frame, "Predict ---->>> " + rps_move, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
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
    classify_rps(model_path)


if __name__ == '__main__':
    main()
