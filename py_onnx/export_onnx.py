import torch
from iresnet import iresnet50

if __name__ == '__main__':
    device = 'cuda'
    model = iresnet50()
    model.load_state_dict(torch.load('ckp/w600k_r50.pth'))
    model.eval().to(device)
    input = torch.randn(1, 3, 112, 112).to(device)
    torch.onnx.export(
            model, input, 'arcface.onnx',
            input_names=['image'],
            output_names=['feature'],
            dynamic_axes={'image': {0: 'batch'}, 'feature': {0: 'batch'}}
    )