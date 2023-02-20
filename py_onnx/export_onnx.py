import torch
from iresnet import iresnet50

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--save_onnx', type=str, help='saved onnx path')

    return parser.parse_args()

def main(opt):
	device = 'cuda'
	model = iresnet50()
	model.load_state_dict(torch.load(opt.weights))
	model.eval().to(device)
	input = torch.randn(1, 3, 112, 112).to(device)
	torch.onnx.export(
            model, input, opt.save_onnx,
            input_names=['image'],
            output_names=['feature'],
            dynamic_axes={'image': {0: 'batch'}, 'feature': {0: 'batch'}}
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)