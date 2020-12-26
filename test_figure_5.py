from renderer import Renderer
import matplotlib.pyplot as plt
import cv2
import os

## ____________________________________________________________
## Target brush stroke
seed = 1      #To have the same image each time
OT_beta = 0   #Parameter for OT weight: 0 => no OT loss
file_path_no_ext = f'./test_images/brush0_OT_beta={OT_beta}'
file_path = file_path_no_ext + '.png'

rd = Renderer(renderer='oilpaintbrush', canvas_color='white')
rd.random_stroke_params(seed=seed)
rd.draw_stroke()
import matplotlib
matplotlib.image.imsave(file_path, rd.foreground) # File is saved in 'test_images'
## ____________________________________________________________


## Tring to optimize from another start stroke with two losses
# Basically a copy of demo.py
import argparse
import torch
import torch.optim as optim
from painter import *

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--img_path', type=str, default=file_path, metavar='str',
                    help='path to test image (default: ./test_images/sunflowers.jpg)')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size ( max(w, h) ) of the canvas for stroke rendering')
parser.add_argument('--max_m_strokes', type=int, default=1, metavar='str',
                    help='max number of strokes (default 500)')
parser.add_argument('--m_grid', type=int, default=1, metavar='N',
                    help='divide an image to m_grid x m_grid patches (default 5)')

## ____________________________________________________________
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=True,
                    help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=OT_beta,
                    help='weight for optimal transportation loss ')
## ____________________________________________________________

parser.add_argument('--net_G', type=str, default='zou-fusion-net', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, or zou-fusion-net (default: zou-fusion-net)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=False,
                    help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    pt.initialize_params()
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_alpha.requires_grad = True
    utils.set_requires_grad(pt.net_G, False)

    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    print('begin to draw...')
    pt.step_id = 0
## ____________________________________________________________
    os.mkdir(file_path_no_ext + '/')
## ____________________________________________________________
    
    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        iters_per_stroke = 20
        if pt.anchor_id == pt.m_strokes_per_block - 1:
            iters_per_stroke = 40
        for i in range(iters_per_stroke):

            pt.optimizer_x.zero_grad()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)
            if args.canvas_color == 'white':
              pt.G_pred_canvas = torch.ones([args.m_grid ** 2, 3, 128, 128]).to(device)
            else:
              pt.G_pred_canvas = torch.zeros(args.m_grid ** 2, 3, 128, 128).to(device)


            pt._forward_pass()            
## ____________________________________________________________
            vis2 = pt._drawing_step_states()
            plt.imsave(file_path_no_ext + '/' + str((i+1)).zfill(4) +
                       '.png', cv2.cvtColor(vis2[:,:,::-1], cv2.COLOR_BGR2RGB))
## ____________________________________________________________


            pt._backward_x()
            pt.optimizer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.step_id += 1

    v = pt.x.detach().cpu().numpy()
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    pt.final_rendered_images = pt._render_on_grids(v_n)
    pt._save_rendered_images()



if __name__ == '__main__':

    pt = Painter(args=args)
    optimize_x(pt)



