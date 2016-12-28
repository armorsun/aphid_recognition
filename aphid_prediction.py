from tools import pyramid
from tools import sliding_window
from model_load import load_model
import argparse
import time
import cv2
import scipy.misc
import numpy as np

ap = argparse.ArgumentParser()
# we can use command: "python filname -i image_path -s 2.5"
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factro size")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

(winW, winH) = (200, 200)
model = load_model('aphids_classifier_models/model_mynet_adagrad.json', 'aphids_classifier_models/model_mynet_adagrad.h5')
file = open('im1_prediction_values', 'w')
n = 1
for resized in pyramid(image, scale=1.5):
    clone = resized.copy()

    for (x, y, window) in sliding_window(resized, step_size=50, window_size=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # print type((x,y,x+winW,y+winH))
        # cv2.imshow("image",cv2.cv.GetSubRect(resized.copy(),(x,y,x+winW,y+winH)))
        # window_img=np.array(np.transpose(resized[x:x+winW, y:winH],(2,0,1)).astype('float32'))/255

        window = cv2.resize(window, (128, 128), interpolation=cv2.INTER_NEAREST)
        window = np.array(np.transpose(window, (2, 0, 1)).astype('float32')) / 255
        window = window[None, ...]
        # print window_img.shape

        # cv2.imshow("image", window_img)
        predictions = model.predict_classes(window, verbose=0)
        prediction_origin_values = model.predict(window, verbose=0)
        file.write(str(predictions[0][0]) + '  ' + str(prediction_origin_values[0][0]) + '\n')
        # print prediction_origin_values
        # print predictions

        if predictions[0][0] == 0:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 1)
            cv2.circle(clone, (x + winW/2, y + winH/2), 10, (255, 0, 0), 2)

            # clone = resized.copy()
            # cv2.rectangle(clone, (x,y), (x+winW, y+ winH),(255,0,0),2)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(10)
    cv2.imwrite('prediction_images/im6/im_200' + str(n) + '.jpg', clone)
    print n
    file.write(str(n) + '')
    n = n + 1
# cv2.destroyAllWindows()
file.close()
