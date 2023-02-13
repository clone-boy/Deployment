import torch
import cv2
from iresnet import iresnet50
import onnxruntime as ort

device = 'cuda'
model = iresnet50()
model.load_state_dict(torch.load('ckp/w600k_r50.pth'))
model.eval().to(device)

image_path = 'img/test.jpg'
img = cv2.imread(image_path)
img = img[:, :, ::-1]
img = torch.Tensor(img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
img.div_(255).sub_(0.5).div_(0.5)


feature = model(img)
print(feature)

ort_session = ort.InferenceSession('ckp/arcface.onnx',providers=['CUDAExecutionProvider'])


feature = ort_session.run(None, { 'image' : img.cpu().numpy() } )
print(feature)

