import torch
import os
from fast_model import TransformerNet
from torchvision import transforms
from utils import load_image, save_image
import argparse

def infer(image, device='cpu', model=None, checkpoint=None):
    if not (checkpoint or model):
        raise "provide correct value for model or checkpoint one at a time"
    elif (model and checkpoint):
        raise "model and checkpoint provided simultaneously, provide on at a time"
    
    device = torch.device(device)

    if checkpoint:
        model = TransformerNet().to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['weights'])
    
    model.eval()
    image = load_image(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))

    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(image)
    out = out.detach().cpu().squeeze(0).clamp(0, 255)
    return out

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    filename = os.path.join(args.save_dir, args.out_name)
    out = infer(args.image, device=args.device, model=None, checkpoint=args.model_path)
    save_image(filename, out)


parser = argparse.ArgumentParser('arguments for styling image')

parser.add_argument('--image', type=str, required=True, help='path to image to be styled')
parser.add_argument('--model_path', type=str, help='path to the saved model')
parser.add_argument('--device', type=str, default='cpu', help='device to be used run model')
parser.add_argument('--save_dir', type=str, default='./output', help='directory to save output images')
parser.add_argument('--out_name', type=str, default='output.jpg', help='name of the output image')


args = parser.parse_args()


if __name__ == '__main__':
    main(args)


