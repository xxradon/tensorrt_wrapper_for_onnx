import onnxruntime
import numpy as np
import time
import cv2 as cv
devices = onnxruntime.get_device()
session = onnxruntime.InferenceSession("./hfnet_github.onnx")
session.get_modelmeta()
first_input_name = session.get_inputs()[0].name
print("hfnet input is {}\n".format(first_input_name))

indata1 = np.ones((1,721,1281,1)).astype(np.float32)
indata1 = indata1 * 128
img = cv.imread("gray_test.bmp")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = img_gray.reshape((1,720,1280,1)).astype(np.float32)
indata1[:,0:720,0:1280,:] = img_gray
results = session.run([], {first_input_name : indata1})

starttime = time.time()
for i in range(1):
    print("index {}\n".format(i))
    results = session.run([], {first_input_name : indata1})
    print("results num is {}".format(len(results)))

# print(results[3].shape)
endtime = time.time()
print((endtime - starttime))
print(results[0])
