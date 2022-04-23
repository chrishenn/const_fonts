from PIL import Image
import numpy as np
from time import time
import os


# recolor a folder of images by (r,g,b) * (shift_kern) element-wise.
# copy recolored images to out_root.
def build_shift(in_root, out_root, shift_kern):
    working = None

    if not os.path.exists(out_root): os.mkdir(out_root)

    r, g, b = shift_kern

    for root, categories, files in os.walk(in_root):
        tic = time()

        for file in files:
            img = Image.open(os.path.join(root, file))
            width, height = img.size

            working = np.array(img).reshape((width, height, 3))


            cha_0 = working[...,0].reshape((width, height, 1))
            cha_0 = cha_0 * r
            cha_0 = np.where(cha_0 > 255, 255, cha_0)
            cha_0.reshape((width,height,1))

            cha_1 = working[...,1].reshape((width, height, 1))
            cha_1 = cha_1 * g
            cha_1 = np.where(cha_1 > 255, 255, cha_1)
            cha_1.reshape((width, height, 1))

            cha_2 = working[...,2].reshape((width, height, 1))
            cha_2 = cha_2 * b
            cha_2 = np.where(cha_2 > 255, 255, cha_2)
            cha_2.reshape((width, height, 1))

            out = Image.fromarray(np.concatenate((cha_0, cha_1, cha_2), axis=2), 'RGB')

            outname = os.path.join(out_root, os.path.basename(root), file)
            if not os.path.exists(os.path.join(out_root, os.path.basename(root))): os.mkdir(os.path.join(out_root, os.path.basename(root)))

            out.save(outname)
            print("Saved to: {} ({:>40.2f})".format(outname, time()-tic))



if __name__ == '__main__':

    shift_kern =    [0,0,1]
    num_cat =       15

    in_root = 'images_out_{}'.format(num_cat)
    out_root = 'images_shift_blue_{}'.format(num_cat)

    build_shift(in_root, out_root, shift_kern)

