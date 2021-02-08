from renderer import Renderer
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from painter import *


# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')

parser.add_argument('--max_m_strokes', type=int, default=1)
parser.add_argument('--m_grid', type=int, default=1)
parser.add_argument('--canvas_color', type=str, default='black')
parser.add_argument('--canvas_size', type=int, default=512)
parser.add_argument('--net_G', type=str, default='zou-fusion-net', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, or zou-fusion-net (default: zou-fusion-net)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=True,
                    help='disable cv2.imshow, for running remotely without x-display')

# Markerpen arguments
parser.add_argument('--renderer', type=str, default='rectangle', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_rectangle', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush)')

# Coefficient arguments
parser.add_argument('--beta_L1', type=float, default=1.0)
parser.add_argument('--beta_ot', type=float, required=True)

parser.add_argument('--iters_per_stroke', type=int, required=True)

args = parser.parse_args()



## ________________________________________________________________________________________
## Ponderation for Optimal transport
OT_beta = args.beta_ot   #Parameter for OT weight: 0 => no OT loss
# NB: markerpen is the default style used

directory = f'./output_test_OT/brush_OT_beta={OT_beta}/'
args.with_ot_loss = ((OT_beta!=0))
args.img_path = directory+'reference.png'

seed = 10      #To have the same image each time

if os.path.exists(f'./output_test_OT/') is False:    
    os.mkdir(f'./output_test_OT/')
if os.path.exists(directory) is False:    
    os.mkdir(directory)
rd = Renderer(renderer=args.renderer, canvas_color='black')
rd.random_stroke_params(seed=seed)
rd.draw_stroke()

plt.imsave(directory+'reference.png', rd.canvas) # File is saved in 'test_images'
plt.imsave(directory+'reference2.png', cv2.cvtColor(rd.canvas, cv2.COLOR_BGR2RGB)) # File is saved in 'test_images'
## ________________________________________________________________________________________
# Decide which device we want to run on


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    pt.initialize_params(seed=seed)
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_alpha.requires_grad = True
    utils.set_requires_grad(pt.net_G, False)

    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    print('begin to draw...')
    pt.step_id = 0

    image_0 = utils.patches2img(pt.G_final_pred_canvas, pt.m_grid).clip(min=0, max=1)
    plt.imsave(directory + 'image_' + str(0).zfill(4) +
                       '.png', cv2.cvtColor(image_0[:,:,::-1], cv2.COLOR_BGR2RGB))

    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        iters_per_stroke = 20
        if pt.anchor_id == pt.m_strokes_per_block - 1:
            iters_per_stroke = 40
        #### to change
        iters_per_stroke = args.iters_per_stroke
        for i in range(iters_per_stroke):

            pt.optimizer_x.zero_grad()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.G_pred_canvas = torch.ones(args.m_grid ** 2, 3, 128, 128).to(device)

            pt._forward_pass()
            image_i = pt._drawing_step_states()
            plt.imsave(directory + 'image_' + str((i+1)).zfill(4) +
                       '.png', cv2.cvtColor(image_i[:,:,::-1], cv2.COLOR_BGR2RGB))
            pt._backward_x()
            pt.optimizer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.step_id += 1

    v = pt.x.detach().cpu().numpy()
    # pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    pt.final_rendered_images = pt._render_on_grids(v_n)
    # pt._save_rendered_images()



if __name__ == '__main__':
    pt = Painter(args=args)
    optimize_x(pt)



