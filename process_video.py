import cv2
from time import sleep, time
from tensorflow.keras.preprocessing import image as image_keras_preprocessing
from tensorflow.keras.models import load_model
from PIL import Image
from PIL.ImageDraw import ImageDraw
from pyglet.window import key
import numpy as np


# -- load keras models
print("Loading Keras: steering model")
model = load_model('saved_model_custom_12.h5')
print("Loading Keras: speed model")
model2 = load_model('saved_model_road_1.h5')
# -------

# -- read video file
cap = cv2.VideoCapture("picam19-02-14-07-46.h264")
ret, frame = cap.read()
# frame dimesions: 1232 x 1640
# ---------------


# -- initialise variables
key_handler = key.KeyStateHandler()
angles_lst = []
color = (0, 200, 0)
lines_angle = [0,0,0]
road_prediction = []

road_value = 3
road_value_prev = 3
# --------------------

while(True):
    # -- read direction key
    key_pressed = cv2.waitKey(3)
    if key_pressed == ord('w'):
        color = (0, 200, 0)
    if key_pressed == ord('a'):
        color = (200, 0, 0)
    if key_pressed == ord('d'):
        color = (0, 0, 200)
    if key_pressed == ord('s'):
        color = (200, 200, 200)
    # print(color)
    # ------------------

    # -- prepare frame image for keras
    im = Image.fromarray(frame)
    im = im.crop((0, 300, 1640, 1032))
    im = im.resize((66, 200), Image.ANTIALIAS)
    img_pred_speed = image_keras_preprocessing.img_to_array(im)
    img_pred_speed = np.expand_dims(img_pred_speed, axis=0)
    draw = ImageDraw(im)
    draw.rectangle((0, 185, 66, 200), fill=color)
    # im.save("steering_frames/" + str(time()) + ".png")
    img_pred = image_keras_preprocessing.img_to_array(im)
    img_pred = np.expand_dims(img_pred, axis=0)
    # ----------------------




    # -- process frame with keras - steernig
    rslt = model.predict(img_pred)
    predicted_angles = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1,
                        0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2,
                        0.3, 0.3, -0.3, 0, -0.3, 0.3]
    angle_float = 0
    # print(rslt[0])
    for idx in range(28):
        angle_float += rslt[0][idx] * predicted_angles[idx]
    if len(angles_lst) > 2:
        angles_lst.pop(0)
    angles_lst.append(round(angle_float, 2))
    final_angle = round(sum(angles_lst) / len(angles_lst), 3)
    # print(final_angle)

    # -- draw direction lines
    lines_angle.append(final_angle)
    lines_angle.pop(0)
    for y in range(480, 50, -20):
        x = 480 - y
        a = (sum(lines_angle) / 3) / 100
        xt = a * x * x
        x11 = 1640 - xt - x * 0.9
        x21 = 0 - xt + x * 0.9
        x = 480 - y - 20
        xt = a * x * x
        x12 = 1640 - xt - x * 0.9
        x22 = 0 - xt + x * 0.9
        cv2.line(frame, (int(x11), y + 752), (int(x12), y + 20 + 752), (200, 80, 80), 15)
        cv2.line(frame, (int(x21), y + 752), (int(x22), y + 20 + 752), (200, 80, 80), 15)
    # ------------------------





    # -- process frame with Keras - speed
    rslt2 = model2.predict(img_pred_speed / 255)
    print("rslt2")
    print(rslt2)
    road_prediction.append(rslt2[0].argmax())
    if len(road_prediction) > 20:
        road_prediction.pop(0)
    # print(road_prediction)
    rv = max(set(road_prediction), key=road_prediction.count)
    if rv != road_value_prev:
        road_value_prev = road_value
    road_value = rv
    if road_value == 3:
        sign_prediction = []

    # print('Road:' + str(road_value))
    speeds = [0.2, 0.4, 0.6, 0.8]
    gl_speed = 0
    for idx in range(4):
        gl_speed += rslt2[0][idx] * speeds[idx]
    print(gl_speed)
    cv2.putText(frame, 'Speed: {} km/h '.format(round((gl_speed * 10),2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (100, 100, 255), 4)
    # --------------------------------






    # -- show frame
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('frame',frame)
    # ---------

    # -- close video
    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
    # -----------

    # -- next frame
    sleep(0.05)
    ret, frame = cap.read()
    # ---------
