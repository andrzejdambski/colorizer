import numpy as np
import glob
from colorizer.preproc.zoom import zoom_on_cat_face

def pil_to_np(pil_image):

    return np.array(pil_image)

def create_local_file_paths_lists():
    l_jpg = []
    l_jpg_name = []
    for n in range(7):
        for file in glob.glob(f'/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_0{n}/*.jpg'):
            l_jpg.append(file)
            l_jpg_name.append(file.replace(f'/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_0{n}/','').replace('.jpg',''))
            
    l_jpg_cat = []
    l_jpg_name_cat = []
    for n in range(7):
        for file in glob.glob(f'/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_0{n}/*.jpg.cat'):
            l_jpg_cat.append(file)
            l_jpg_name_cat.append(file.replace(f'/home/andrzej/code/andrzejdambski/Projet_colorisation/2/CAT_0{n}/','').replace('.jpg.cat',''))
            
    return l_jpg,l_jpg_name,l_jpg_cat,l_jpg_name_cat

def create_list_np():
    
    l_jpg,l_jpg_name,l_jpg_cat,l_jpg_name_cat = create_local_file_paths_lists()
    
    l_jpg_cat = [file+'.cat' for file in l_jpg]

    list_np = []
    
    for cat_image,file_cat in list(zip(l_jpg,l_jpg_cat)):
        pil_image = zoom_on_cat_face(cat_image,file_cat)
        if pil_image != None:
            if pil_image.size==(64,64):
                list_np.append(pil_to_np(pil_image))

    X = np.stack(list_np)
    return X

def create_list():
    l = []
    return l