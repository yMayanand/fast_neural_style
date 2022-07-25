import os
import gc
import argparse

from utils import *
from vgg_model import VGG16
from loss import ContentLoss, StyleLoss, TotalVariationLoss
from fast_model import TransformerNet
from stylize import infer
import random
from get_resources import download_and_extract

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

URL = 'http://images.cocodataset.org/zips/val2017.zip'
fname = URL.split('/')[-1]
if not os.path.exists(fname):
    download_and_extract(URL, fname)

parser = argparse.ArgumentParser('arguments for training')

parser.add_argument('--data_dir', type=str, default='./val2017', help='path to image directory')
parser.add_argument('--style_image', type=str, help='path to style image')
parser.add_argument('--exp_name', type=str, help='name of experiment that appears in tensorboard')
parser.add_argument('--checkpoint', type=str, help='path of directory to save models')

parser.add_argument('--lr', type=float, default=5e0, help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of optimization steps')
parser.add_argument('--log_interval', type=int,
                    default=10, help='logging interval')
parser.add_argument('--resume', type=str,
                    default=None, help='path to the saved model')


parser.add_argument('--style_size', type=int,
                    default=None, help='style image size')
parser.add_argument('--content_size', type=int,
                    default=256, help='content image size')

parser.add_argument('--content_weight', type=float,
                    default=1e5, help='content weight for loss')
parser.add_argument('--style_weight', type=float,
                    default=3e4, help='style weight for loss')
parser.add_argument('--tv_weight', type=float,
                    default=1e0, help='total variation weight for loss')
parser.add_argument('--bs', type=int,
                    default=32, help='batch size for training')
parser.add_argument('--workers', type=int,
                    default=1, help='number of worker in dataloader')
parser.add_argument('--log_image', type=int,
                    default=2, help='logs  images after every n batches')


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    writer = SummaryWriter(f"runs/{args.exp_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path to images
    style_img_path = args.style_image
    
    # list all files
    files = os.listdir(args.data_dir)

    # train dataset
    train_ds = Dataset(
        files, 
        args.data_dir, 
        shape=args.content_size
    )

    # train dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, 
        batch_size=args.bs, 
        shuffle=True,
        num_workers=args.workers
    )

    if args.style_size is not None:
        style_shape = (args.style_size, args.style_size)
    else:
        style_shape = None

    style_image = load_image(
        style_img_path,
        shape=style_shape
    )


    # preprocess image
    style_tensor = transform(style_image).to(device).unsqueeze(0)

    # model
    vgg = VGG16(requires_grad=False).to(device)

    # style model
    transformer = TransformerNet().to(device)

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        transformer.load_state_dict(state_dict['weights'])

    # optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)


    vgg.eval()
    target_style_features = vgg(style_tensor)
    
    # loss functions
    content_loss = ContentLoss(content_weight=args.content_weight)
    style_loss = StyleLoss(style_weight=args.style_weight)
    tv_loss = TotalVariationLoss(tv_weight=args.tv_weight)

    # recording meter
    c_loss_meter = AverageMeter()
    s_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()
    total_loss = AverageMeter()

    for epoch in range(args.epochs):
        for batch, data in enumerate(train_dl):
            batch_size = data.shape[0]

            target_style_features_ = []
            for i in target_style_features:
                target_style_features_.append(torch.tile(i, (batch_size, 1, 1, 1)).detach())

            data = data.to(device)

            vgg.eval()
            target_content_features = vgg(normalize_batch(data)).relu2_2

            transformer.train()
            image = transformer(data)
            image = normalize_batch(image)
            image_style_features = vgg(image)
            image_content_features = image_style_features.relu2_2

            s_l = 0
            c_l = 0
            t_l = 0

            for y, x in zip(target_style_features_, image_style_features):
                s_l += style_loss(y, x)

            c_l = content_loss(target_content_features, image_content_features)

            #t_l = tv_loss(image)
            

            loss = c_l + s_l #+ t_l

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            c_loss_meter.update(c_l.item())
            s_loss_meter.update(s_l.item())
            #tv_loss_meter.update(t_l.item())
            total_loss.update(loss.item())

            del target_content_features, image, image_content_features, image_style_features, loss
            gc.collect()
            torch.cuda.empty_cache()

            global_step = (epoch * len(train_dl)) + batch

            if global_step % args.log_image == (args.log_image - 1):
                idx = random.randint(0, (len(files) - 1))
                image_path = os.path.join(args.data_dir, files[idx])
                result = infer(image_path, device=str(device), model=transformer)
                writer.add_image('debug_images', result.to(torch.uint8), global_step)

            writer.add_scalar('loss/style_loss', s_loss_meter.val, global_step)
            writer.add_scalar('loss/content_loss', c_loss_meter.val, global_step)
            #writer.add_scalar('loss/tv_loss', tv_loss_meter.val, global_step)
            writer.add_scalar('loss/total_loss', total_loss.val, global_step)


        if (epoch % args.log_interval) == (args.log_interval -1):
            print(f"epoch: {epoch:04}, \
            variation_loss: {tv_loss_meter.avg:12.4f}, \
            style_loss: {s_loss_meter.avg:12.4f}, \
            content_loss: {c_loss_meter.avg:12.4f}, \
            total_loss: {total_loss.avg:12.4f}")
            save_model(transformer, args.checkpoint)


args = parser.parse_args()

if __name__ == "__main__":
    main(args)
