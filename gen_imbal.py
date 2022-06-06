import numpy as np
import random
import os
from PIL import Image
import PIL.ImageOps
import time
import torchvision.transforms.functional as func

# TODO: have to build a new templ object for each template; fix

class templ:
    """
    letter/number template class, composed of template rectangles specified in templ_dict
    templ_dict keyed by unique ints (that are not important, only must be unique), and map to:
    {key: [location (x, y) tuple, orientation (degrees counterclockwise from vertical, ratio of rectangle width/height]}
    """

    def __init__(self, letter, templ_dict):
        self.templ_dict = templ_dict
        self.letter = letter


    # will populate the template with num_build new constructed images
    # with variance +- ratio_radius/100 in rectangle ratios
    # hue balance can be shifted green<->red using hue_factor.
    # templ_id used in the filename. Pass a different templ_id for each templ used in construction
    def build_imbal(self, num_build, ratio_radius, templ_id, out_size, hue_factor, in_folder_root, out_folder_root):
        ratio_folder = ''

        for i in range(num_build):
            new_im = Image.new(mode='RGBA', size=(4000, 4000), color=(255, 255, 255, 0))

            for seg in self.templ_dict:
                im = None
                center_loc, orient, ratio, max_height = self.templ_dict[seg]

                # will notify if np.random generated an invalid path
                try:
                    # randomize the ratio folder this image is drawn from
                    ratio += (np.random.randint(-ratio_radius, ratio_radius) / 100)
                    ratio_folder = in_folder_root + 'ratio_{:0.2f}'.format(ratio)
                    im = Image.open(ratio_folder +'/'+ random.choice(os.listdir(ratio_folder)))
                except:
                    print("tried to open folder ", ratio_folder, " but it didn't exist. Generate more images.")

                # max dim is always height
                if im.width > im.height: im = im.rotate(90, expand=True, fillcolor=(255, 255, 255, 0))

                # resize to conform to max_height
                im = im.resize(( int((max_height / im.height) * im.width), max_height) )

                # rotate to 'orient' angle
                im = im.rotate(orient, expand=True, fillcolor=(255, 255, 255, 0))

                # translate coordinates from center_img to top left corner
                loc = (int(center_loc[0] - im.width/2), int(center_loc[1] - im.height/2))

                new_im.paste(im, loc, mask=im)


            #flatten to RGB
            new_im_RGB = new_im.convert('RGB')
            new_im_inv = PIL.ImageOps.invert(new_im_RGB)

            #padding, resizing to out_size
            left, upper, right, lower = new_im_inv.getbbox()
            new_im = new_im_RGB.crop((left-100, upper-100,right+100, lower+100))
            new_im = new_im.resize(out_size)

            #recolor
            new_im = func.adjust_hue(new_im, hue_factor)

            #save, folder by letter-subdir
            if not os.path.exists(out_folder_root): os.mkdir(out_folder_root)
            if not os.path.exists(out_folder_root + letter): os.mkdir(out_folder_root + letter)

            new_im.save( out_folder_root + '/{0}/{1:016d}.png'.format(letter,templ_id) )

            templ_id+=1

        return templ_id



if __name__ == '__main__':

    center_x, center_y = 2000, 2000

    # dict = { id:[center location, orientation, rectangle ratio, rectangle max_height] }
    templ_dict_A ={2:[(center_x - 180, center_y + 250  ), -15, 0.35,   260],  # up_left_1
                    3:[(center_x - 110, center_y        ), -15, 0.35,   260],  # up_left_2
                    4:[(center_x - 50,  center_y - 250  ), -15, 0.35,   260],  # up_left_3

                    7:[(center_x + 70,  center_y - 250  ),  15, 0.35,   260],  # down_right_1
                    6:[(center_x + 130, center_y        ),  15, 0.35,   260],  # down_right_2
                    5:[(center_x + 200, center_y + 250  ),  15, 0.35,   260],  # down_right_3

                    1:[(center_x,       center_y + 80   ),  90, 0.40,   200]}  # hori_center

    templ_dict_B = {0:[(center_x,       center_y        ),  90, 0.45,   180],  # hori_center
                    1:[(center_x,       center_y - 400  ),  90, 0.45,   180],  # hori_top
                    2:[(center_x,       center_y + 400  ),  90, 0.45,   180],  # hori_bot

                    3:[(center_x + 160, center_y - 340  ),  45, 0.40,   180],  # down_top
                    7:[(center_x + 240, center_y - 220  ),   0, 0.40,   160],  # vert_top
                    4:[(center_x + 160, center_y - 80   ), -45, 0.40,   180],  # up_top

                    5:[(center_x + 160, center_y + 80   ),  45, 0.40,   180],  # down_bot
                    8:[(center_x + 240, center_y + 220  ),   0, 0.40,   160],  # vert_bot
                    6:[(center_x + 160, center_y + 340  ), -45, 0.40,   180],  # up_bot

                    9:[(center_x - 100,  center_y + 270  ),   0, 0.30,   270],   # vert_bar_bot
                    10:[(center_x - 100, center_y        ),   0, 0.30,   270],    # vert_bar_middle
                    11:[(center_x - 100, center_y - 270  ),   0, 0.30,   270],   # vert_bar_top

                    }

    templ_dict_E = {0:[(center_x,       center_y        ), 90, 0.30,    450],  # hori_center
                    1:[(center_x,       center_y - 400  ), 90, 0.30,    450],  # hori_top
                    2:[(center_x,       center_y + 400  ), 90, 0.30,    450],  # hori_bot

                    9:[(center_x - 200,  center_y + 270  ),   0, 0.30,   270],   # vert_bar_bot
                    10:[(center_x - 200, center_y        ),   0, 0.30,   270],    # vert_bar_middle
                    11:[(center_x - 200, center_y - 270  ),   0, 0.30,   270],   # vert_bar_top
                    }

    templ_dict_F = {0:[(center_x,       center_y        ), 90, 0.20,    450],  # hori_center
                    1:[(center_x,       center_y - 400  ), 90, 0.20,    450],  # hori_top

                    2:[(center_x - 200,  center_y + 270  ),   0, 0.25,   270],   # vert_bar_bot
                    3:[(center_x - 200, center_y        ),   0, 0.25,   270],    # vert_bar_middle
                    4:[(center_x - 200, center_y - 270  ),   0, 0.25,   270],   # vert_bar_top
                    }

    templ_dict_H = {0:[(center_x,       center_y        ), 90, 0.25,    350],  # hori_center

                    1: [(center_x - 200, center_y + 270 ), 0, 0.35,     270],    # vert_bar_bot_left
                    2: [(center_x - 200, center_y       ), 0, 0.35,     270],     # vert_bar_middle_lfet
                    3: [(center_x - 200, center_y - 270 ), 0, 0.35,   270],      # vert_bar_top_lfet

                    4: [(center_x + 200, center_y + 270 ), 0, 0.35,    270],  # vert_bar_bot_right
                    5: [(center_x + 200, center_y      ), 0, 0.35,     270],  # vert_bar_middle_right
                    6: [(center_x + 200, center_y - 270), 0, 0.35,     270],  # vert_bar_top_right

                    }

    templ_dict_I = {0:[(center_x - 90, center_y - 450  ), 90, 0.35,    210],  # hori_top_left
                    1:[(center_x + 90, center_y - 450  ), 90, 0.35,    210],  # hori_top_right

                    2:[(center_x - 90, center_y + 450  ), 90, 0.35,    210],  # hori_bot_left
                    3:[(center_x + 90, center_y + 450  ), 90, 0.35,    210],  # hori_bot_right

                    4:[(center_x,       center_y + 270  ),   0, 0.35,   280],   # vert_bar_bot
                    5:[(center_x,       center_y        ),   0, 0.35,   280],    # vert_bar_middle
                    6:[(center_x,       center_y - 270  ),   0, 0.35,   280],   # vert_bar_top
                    }

    templ_dict_K = {5:[(center_x + 40,  center_y - 90    ), -45, 0.20,    220],  # up_top_1
                    6:[(center_x + 210, center_y - 260  ), -45, 0.20,   310],  # up_top_2

                    7:[(center_x + 40,  center_y + 90 ),  40, 0.20,    250],  # down_bot_1
                    8:[(center_x + 210, center_y + 260  ), 40, 0.20,    300],  # down_bot_2

                    1:[(center_x - 100,  center_y + 270  ),   0, 0.35,  270],   # vert_bar_bot
                    2:[(center_x - 100,  center_y        ),   0, 0.35,   270],    # vert_bar_middle
                    3:[(center_x - 100,  center_y - 270  ),   0, 0.35,   270],   # vert_bar_top
                    }  # vert_bar

    templ_dict_L = {1:[(center_x + 30, center_y + 420  ),  90, 0.30,    270],  # hori_bot_1
                    5:[(center_x + 270, center_y + 420  ),  90, 0.30,   250],  # hori_bot_2

                    4:[(center_x - 100,  center_y + 270  ),   0, 0.30,  280],   # vert_bar_bot
                    2:[(center_x - 100,  center_y        ),   0, 0.30,   280],    # vert_bar_middle
                    3:[(center_x - 100,  center_y - 270  ),   0, 0.30,   280],   # vert_bar_top
                    }

    templ_dict_T = {1:[(center_x - 170, center_y - 420  ),   90, 0.30,  180],   # hori_top_left
                    2:[(center_x,       center_y - 420  ),   90, 0.30,   180],    # hori_top_middle
                    3:[(center_x + 170, center_y - 420  ),   90, 0.30,   185],   # hori_top_right

                    4:[(center_x,    center_y + 270     ),   0, 0.30,  290],   # vert_bar_bot
                    5:[(center_x,    center_y           ),   0, 0.30,   290],    # vert_bar_middle
                    6:[(center_x,    center_y - 270     ),   0, 0.30,   290],   # vert_bar_top
                    }

    templ_dict_V = {4:[(center_x - 180,     center_y - 250  ), 15, 0.35,   270],  # down_left_1
                    3:[(center_x - 110,     center_y        ), 15, 0.35,   270],  # down_left_2
                    2:[(center_x - 50,      center_y + 250   ), 15, 0.35,  270],  # down_left_3


                    7:[(center_x + 40,  center_y + 250  ),  -15, 0.35,   270],  # up_right_1
                    6:[(center_x + 100, center_y        ),  -15, 0.35,   270],  # up_right_2
                    5:[(center_x + 170, center_y - 250  ),  -15, 0.35,   270]    # up_right_3
                   }

    templ_dict_X = {4:[(center_x - 100,     center_y - 250  ), 20, 0.20,   300],  # down_left_1
                    3:[(center_x,           center_y        ), 20, 0.20,   300],  # down_left_2
                    2:[(center_x + 100,      center_y + 250   ), 20, 0.20,  300],  # down_left_3


                    7:[(center_x - 100,      center_y + 250  ),  -20, 0.20,   300],  # up_right_1
                    6:[(center_x,           center_y        ),  -20, 0.20,   300],  # up_right_2
                    5:[(center_x + 100,      center_y - 250  ),  -20, 0.20,   300]    # up_right_3
                   }

    templ_dict_Y = {0:[(center_x - 50,  center_y - 170  ),  20, 0.20,   230],    # down_left
                    2:[(center_x + 50,  center_y - 170  ), -20, 0.20,   230],    # up_right

                    4:[(center_x,       center_y        ),   0, 0.25,   160],   # vert_bar_bot
                    6:[(center_x,       center_y + 150  ),   0, 0.25,   180],   # vert_bar_top
                    }


    alphabet = {'A':templ_dict_A, 'B':templ_dict_B, 'E':templ_dict_E, 'F':templ_dict_F, 'H':templ_dict_H, 'I':templ_dict_I,
                'K':templ_dict_K, 'L':templ_dict_L, 'T':templ_dict_T, 'V':templ_dict_V, 'X':templ_dict_X, 'Y':templ_dict_Y}

    templ_id = 10

    for letter in alphabet:
        tic = time.time()
        tmp = templ(letter, alphabet[letter])

        num_percat =        15
        hue_shift =         -0.5
        out_size =          (224,224)
        out_folder_root =   'images_imbal_' + str(num_percat) + '_' + str(hue_shift) + '/'
        in_folder_root =    'images_sorted/'

        templ_id = tmp.build_imbal(num_build=num_percat, ratio_radius=5, templ_id=templ_id, out_size=out_size, hue_factor=hue_shift,
                             in_folder_root=in_folder_root, out_folder_root=out_folder_root)
        templ_id+=10
        print("template built for: {} {:0.2f}s'".format(letter, time.time() - tic))
