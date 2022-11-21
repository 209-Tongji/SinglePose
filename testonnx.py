import onnx
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import get_max_pred, draw_origin_joints

model = onnx.load("./ncnnModels/litehrnet/litehrnet_coco_best.onnx")
onnx.checker.check_model(model)

session = ort.InferenceSession("./ncnnModels/litehrnet/litehrnet_coco_best.onnx")

image = Image.open("bbox0.png", mode='r').convert('RGB')
scale_w = image.width / 192.0
scale_h = image.height / 256.0
resize = transforms.Resize((256, 192))
to_tensor = transforms.ToTensor()
img = resize(image)
img = to_tensor(img)
img[0].add_(-0.406)
img[1].add_(-0.457)
img[2].add_(-0.480)
x = img.unsqueeze(0)
x = x.numpy()
print(x.shape)
#x = np.random.randn(1,3,256,192).astype(np.float32)

outputs = session.run(None, input_feed = {'input': x})

result, _ = get_max_pred(outputs[0][0])

for i in range(len(result)):
    result[i][0] *= 4 * scale_w
    result[i][1] *= 4 * scale_h

draw_origin_joints(image, result, output="onxxres.png")
print(result)
print(outputs[0].shape)
print(outputs[0])