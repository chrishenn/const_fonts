from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pickle
import time
from os import path, mkdir
from PIL import Image



# build a COCO object with index from pickled coco.p OR path to annotation .json (ann_path)
def build_coco(ann_path):
    if path.exists("coco.p"):
        tic = time.time()
        coco = pickle.load(open("coco.p", "rb"))
        print('Load coco object from pickle done: {:0.2f}s'.format(time.time() - tic))
    else:
        coco = COCO(ann_path)
        pickle.dump( coco, open( "coco.p", "wb" ) )

    return coco

# cut out segmented parts of images, sort by ratio into folders (one folder for each 0.01 range of ratios)
# images saved in RGBA .png; backgrounds are transparent; cropped to bbox of cutout
def sort_by_ratio(coco, num_imgs_in, out_path):
    tic = time.time()

    all_ann_ids = coco.getAnnIds(iscrowd=False)
    all_anns = coco.loadAnns(all_ann_ids)

    imgobjs = coco.loadImgs(ids=[ann['image_id'] for ann in all_anns])[:num_imgs_in]

    count = 0
    for obj in imgobjs:

        try:
            fname = 'data/images_2017/' + obj['file_name']
            I = io.imread(fname)
        except: continue

        ann_ids = coco.getAnnIds(imgIds=obj['id'], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x,y,width,height = ann['bbox']

            if height > 0:
                ratio = width / height
            else: continue

            if ratio > 1:
                ratio = 1.0 / ratio

            if width > 20 and height > 20 and (0.14 <= ratio <= 0.56):

                try:
                    ratio_str = out_path + 'ratio_{:0.2f}'.format(ratio)
                    if not path.exists(out_path): mkdir(out_path)
                    if not path.exists(ratio_str): mkdir(ratio_str)

                    mask = coco.annToMask(ann)*255

                    image = I.copy()
                    new = np.zeros_like(image)

                    for rgb in 0,1,2:
                        new[:,:,rgb] = np.bitwise_and(image[:,:,rgb], mask)


                    npad = ((0,0),(0,0),(0,1))
                    new = np.pad(new, pad_width=npad, mode='constant', constant_values=255)
    
                    new[:, :, 3] = np.bitwise_and(new[:, :, 3], mask)

                    cropped_im = Image.fromarray(new).crop((x, y, x+width, y+height))

                    fname = ratio_str + '/fig_{:016d}'.format(count) + '.png'
                    cropped_im.save(fname)

                    print("Saved: ", fname)
                    count+=1
                except: continue

    print('sorted by ratio done: {:0.2f}s'.format(time.time() - tic))

# build COCO obj from ann_path; extract segmented objects and sort by aspect ratio (width / height)
if __name__ == '__main__':
    num_imgs_in = 10000
    out_path = 'images_sorted/'
    ann_path = 'data/annotations_2017/instances_train2017.json'

    coco = build_coco(ann_path)
    sort_by_ratio(coco, num_imgs_in, out_path)