import imageio
from PIL import Image, ImageFont, ImageDraw
import glob
import os
import numpy as np
import argparse
from tqdm import *

parser = argparse.ArgumentParser(description='Make GIF')
parser.add_argument('input_dir',
                    help='Directory of images to make GIF')
parser.add_argument('output_path',
                    help='Output path of GIF')
args = parser.parse_args()


with imageio.get_writer(args.output_path, mode='I') as writer:
    for img_path in tqdm(sorted(glob.glob( os.path.join(args.input_dir, '*.jpg'))), leave=True):
        img = Image.open(img_path).convert('RGBA')
        txt = Image.new('RGBA', img.size, (255,255,255,0))
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("DejaVuSerif-Bold.ttf", 24)
        title = 'Epoch_' + img_path.split('_')[1]
        w, h = draw.textsize(title, font)
        draw.rectangle([10,10,10+w,10+h], fill=(255, 255, 255, 160))
        draw.text((10,10), title, font=font, fill=(255,0,0,255))
        out = Image.alpha_composite(img, txt)
        writer.append_data(np.array(out))

