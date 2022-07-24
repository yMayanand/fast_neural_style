import torch
from fast_model import TransformerNet
from utils import load_image, preprocess_image

def infer(model, image, device, checkpoint=None):
    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['weights'])
    
    model.eval()
    image = load_image(image)
    image = preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(image)
    out = out.detach().cpu().squeeze(0).clamp(0, 255)
    return out




