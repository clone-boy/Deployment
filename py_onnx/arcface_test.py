import torch
import cv2
from iresnet import iresnet50
import onnxruntime as ort

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--save_onnx', type=str, help='saved onnx path')
    parser.add_argument('--img_path', type=str, help='img path')

    return parser.parse_args()

def main(opt):
    device = 'cuda'
    model = iresnet50()
    model.load_state_dict(torch.load(opt.weights))
    model.eval().to(device)

    image_path = opt.img_path
    img = cv2.imread(image_path)
    img = img[:, :, ::-1]
    img = torch.Tensor(img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    img.div_(255).sub_(0.5).div_(0.5)


    feature = model(img)
    print(feature)

    ort_session = ort.InferenceSession(opt.save_onnx,providers=['CUDAExecutionProvider'])
    feature = ort_session.run(None, { 'image' : img.cpu().numpy() } )
    print(feature)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
