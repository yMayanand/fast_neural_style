import torch
from fast_model import TransformerNet
from utils import load_image, preprocess_image

def infer(model, image, checkpoint=None):
    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['weights'])
    
    model.eval()
    image = load_image(image)
    image = preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        out = model(image)
    return out.detach().cpu().squeeze(0)




