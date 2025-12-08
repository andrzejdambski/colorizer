import cv2 as cv
from google.cloud import storage
import os
from PIL import Image
import io
import numpy as np

def getting_file_names(file_type='jpg'):
    '''
    cette fonction recupere le nom des fichiers dans le bucket
    Requiere le package:
    from google.cloud import storage

    Returns un fichier avec les file names en jpg
    '''
    client = storage.Client()
    # bucket = client.bucket("colorizer")
    bucket = client.bucket("nina-cats-data")

    file_names = []

    for blob in bucket.list_blobs():
        if blob.name.endswith(f'.{file_type}'):
            file_names.append(blob.name)

    return file_names

def download_file_form_bucket():
    '''
    Telecharge les fichiers depuis le seau
    '''
    client = storage.Client()
    bucket = client.bucket("colorizer")

    file_names = []

    for blob in bucket.list_blobs():
        if blob.name.endswith('.jpg'):
            file_names.append(blob.name)

    for i in range(len(file_names)):
        bloby = bucket.blob(file_names[i])
        bloby.download_to_filename(file_names[i])

    return None



def vectorise_jpg():
    '''
    vectorise une image jpg, retournant une liste de numpy array.
    Faut lui donner un path pour qu'il aille chercher le fichier
    Necessite cv2
    import cv2 as cv
    '''
    root = os.getcwd()

    imgpath=os.path.join('/Users/constantindorleans/Downloads/00000001_000.jpg')

    img=cv.imread(imgpath)

    return img


def vectorise_file_from_bucket():
    '''
    Pioche dans le bucket le nom des fichiers, et les vectorise un par un
    appendant a une liste.
    '''
    client = storage.Client()
    bucket = client.bucket("colorizer")

    file_names = []
    vectorised_images = []

    for blob in bucket.list_blobs():
        if blob.name.endswith('.jpg'):
            file_names.append(blob.name)


    for i in range(len(file_names)-9961):
        bloby = bucket.blob(file_names[i])

        img_bytes = bloby.download_as_bytes()
        image = Image.open(io.BytesIO(img_bytes))
        arr = np.array(image)
        vectorised_images.append(arr)

    return vectorised_images

    # img_bytes = bloby.download_as_bytes()
    # image = Image.open(io.BytesIO(img_bytes))
    # arr = np.array(image)
