import math
from PIL import Image
# cat_image = '/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_00/00000001_005.jpg'
# file_cat = '/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_00/00000001_005.jpg.cat'

def zoom_on_cat_face(cat_image,file_cat,data_augmentation=False):

    """Takes the file paths for a .jpg and the .cat associated,
    scales, rotates and crops the image to a 64x64 image of a centered cat's head.
    The output is a Image.PIL

    Input: cat_image : file path to the .jpg of a cat
            file_cat : file path to the .cat associated
    Returns:
        Image.PIL: 64x64 image of a centered cat's head
    """    
    cat_image = Image.open(cat_image)
    cat_cat = open(file_cat,'r')
    cat_cat.seek(0)
    cat_cat = cat_cat.read()
    cat_leye = cat_cat.split()[1:3]
    cat_reye = cat_cat.split()[3:5]
    cat_leye = [int(coor) for coor in cat_leye]
    cat_reye = [int(coor) for coor in cat_reye]
    
    # output position of the left eye
    leye_loc = [18,30]

    #scale
    distance_eyes = math.sqrt((cat_reye[1]-cat_leye[1])**2+(cat_reye[0]-cat_leye[0])**2)
    distance_output = 64 - 2*leye_loc[0]
    scale = distance_eyes/distance_output
    a = scale
    b = 0
    c = ((1-scale)*cat_image.size[0])/2
    d = 0
    e = scale
    f = ((1-scale)*cat_image.size[1])/2
    cat_s = cat_image.transform(cat_image.size, Image.AFFINE, (a,b,c*(1),d,e,f*(1)),Image.Resampling.BILINEAR)

    #rotation
    
    if cat_reye[1]>cat_leye[1]:
        sign = 1
    else:
        sign = -1
    rotation = math.atan((cat_reye[1]-cat_leye[1])/(cat_reye[0]-cat_leye[0]))*(180/math.pi)
    
    new_leye_center = (cat_image.size[0]/2 - 1/scale*(cat_image.size[0]/2 - cat_leye[0]),cat_image.size[1]/2 - 1/scale*(cat_image.size[1]/2 - cat_leye[1]))
    
    cat_r = cat_s.rotate(sign*rotation,center=new_leye_center,resample = Image.Resampling.BILINEAR)

    #crop
    x_crop = cat_leye[0]-leye_loc[0]
    y_crop = cat_leye[1]-leye_loc[1]

    center_x = cat_image.size[0]/2
    center_y = cat_image.size[1]/2
    cat_c = cat_r.crop((new_leye_center[0]-leye_loc[0],new_leye_center[1]-leye_loc[1],\
                        new_leye_center[0]+(64-leye_loc[0]),new_leye_center[1]+(64-leye_loc[1])))
    
    return cat_c
