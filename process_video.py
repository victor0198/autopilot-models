import cv2
from time import sleep, time
from tensorflow.keras.preprocessing import image as image_keras_preprocessing
from tensorflow.keras.models import load_model
from PIL import Image
from PIL.ImageDraw import ImageDraw
from pyglet.window import key
import numpy as np
import tensorflow as tf


# -- load keras models
print("Loading Keras: steering model")
model = load_model('saved_model_custom_12.h5')
print("Loading Keras: speed model")
model2 = load_model('road_1.h5')
# -------

# -- loading Tensorflow model
print("Loading Tensorflow: sign detection model")
#MODEL_NAME = 'gym_duckietown/object_detection/inference_graph1'
PATH_TO_FROZEN_GRAPH = 'detect.tflite' # MODEL_NAME + '/detect.tflite'
PATH_TO_LABELS = 'labelmap.pbtxt' #'gym_duckietown/object_detection/inference_graph1/labelmap.pbtxt'

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path=PATH_TO_FROZEN_GRAPH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_height = input_details[0]['shape'][1]
img_width = input_details[0]['shape'][2]
print("this: ",img_height, img_width)
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
# ------------------------


# -- read video file
cap = cv2.VideoCapture("picam2020_10_5_16_28_30.h264")
ret, frame = cap.read()
# frame dimesions: 1232 x 1640
# ---------------


# -- initialise variables
key_handler = key.KeyStateHandler()
angles_lst = []
color = (0, 200, 0)
lines_angle = [0,0,0]
# road_prediction = []

road_value = 3
road_value_prev = 3

wheels = 0
engine = 0

sign_prediction = []
# --------------------

# <
step = 2300
# >

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

    if wheels != 0 and step<0:
        ims.save("steering_speed/" + wheels + "." + str(engine) + "." + str(time()) + ".png")

    # -- prepare frame image for keras

    # frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ims = Image.fromarray(frame2)
    #ims.save("steering_frames/" + str(time()) + ".png")

    im = Image.fromarray(frame)
    im = im.crop((0, 300, 1640, 1032))
    im = im.resize((66, 200), Image.ANTIALIAS)
    draw = ImageDraw(im)
    draw.rectangle((0, 185, 66, 200), fill=color)
    # im.save("steering_frames/" + str(time()) + ".png")
    img_pred = image_keras_preprocessing.img_to_array(im)
    img_pred = np.expand_dims(img_pred, axis=0)

    im_speed = Image.fromarray(frame)
    im_speed = im_speed.crop((0, 320, 1640, 1120))
    im_speed = im_speed.resize((130, 245), Image.ANTIALIAS)
    # im_speed.save("steering_frames/zzz" + str(time()) + ".png")
    img_pred_speed = image_keras_preprocessing.img_to_array(im_speed)
    img_pred_speed = np.expand_dims(img_pred_speed, axis=0)
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
    rslt2 = model2.predict(img_pred_speed) # / 255)
    # print("rslt2")
    # print(rslt2)
    # road_prediction.append(rslt2[0].argmax())
    # if len(road_prediction) > 20:
    #     road_prediction.pop(0)
    # # print(road_prediction)
    # rv = max(set(road_prediction), key=road_prediction.count)
    # if rv != road_value_prev:
    #     road_value_prev = road_value
    # road_value = rv
    # if road_value == 3:
    #     sign_prediction = []

    # print('Road:' + str(road_value))
    speeds = [0.7, 0.4, 0.6, 0.3, 0.5, 0.2, 0.0, 0.1, 0.8, 0.9]
    statuses = ["car", "cross", "curve", "downhill", "intersection", "traffic light", "complete stop",
                "stop line", "straight road", "uphill"]
    gl_speed = 0
    for idx in range(10):
        gl_speed += rslt2[0][idx] * speeds[idx]
        if rslt2[0][idx] > 0.9:
            status = statuses[idx]
    #print(gl_speed)
    cv2.putText(frame, 'Speed: {} km/h '.format(round((gl_speed * 10),2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (100, 100, 255), 4)
    cv2.putText(frame, 'Status: {}'.format(status), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (100, 100, 255), 4)
    # --------------------------------




    # -- object detection
    image = np.array(frame)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image, (img_width, img_height))
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # Image.fromarray(image_resized).save("steering_frames/" + str(time()) + ".png")
    input_data = np.expand_dims(image_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i, b in enumerate(boxes):
        if scores[i] >= 0.5:
            apx_dist = (0.2 * 3.04) / boxes[i][1]

    for i in range(len(scores)):
        if (scores[i] >= 0.7) and (scores[i] <= 1.0):
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3]) * imW))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelsize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelsize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelsize[1] - 10),
                          (xmin + labelsize[0], label_ymin + baseline - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if apx_dist < 0.5:
                # print(label)
                if 'pedestrian_cross' in label:
                    sign_prediction.append(1)
                    cv2.putText(frame, "CROSS", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                if 'stop' in label:
                    sign_prediction.append(2)
                    cv2.putText(frame, "STOP", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                if 'priority' in label:
                    sign_prediction.append(3)
                    cv2.putText(frame, "SLOW DOWN", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                if len(sign_prediction) > 3:
                    sign_prediction.pop(0)
                print(sign_prediction)
                if len(sign_prediction) == 0:
                    sign_prediction.append(0)
                sign_value = max(set(sign_prediction), key=sign_prediction.count)
                print('Sign:' + str(sign_value))

    # ----------------





    # -- show frame
    #print("-")
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('frame',frame)
    # print("-")
    # ---------

    if step<0:
        wheels = input()
        engine = input()
    

    # -- close video
    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
    # -----------

    # -- next frame
    sleep(0.001)
    ret, frame = cap.read()
    # ---------
    step -=1
