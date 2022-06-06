# Constant Invariate Fonts

Contribution to Object-Oriented Deep Learning (OODL) project. Advisor: [Qianli Liao](https://cbmm.mit.edu/about/people/liao)

Project to generate difficult-to-learn visual datasets; evaluate traditional convnets; and motivate novel network architectures. We manipulate segments of images from the cocodataset.org dataset (2017) to construct capital letters from image segments. We can skew or recolor these composite images as needed.

&emsp;

## coco_subimg.py
Cuts out annotated segmented-objects from coco imgs; sorts subimages by aspect ratio (width/heigh) of segment's bounding box.

## build_templ.py
Populates baked-in templates with subimages, having been presorted by aspect ratio. Templates include 12-letter subset of uppercase alphabet (A,B,E,F,H,I,K,L,T,V,X,Y).

## gen_imbal.py
Populates templates with subimages, skews colors towards specified values.

## recolor.py
Recolors folder of images by (r,g,b)*shift_kern element-wise. Copies new color-shifted images to specified folder.

