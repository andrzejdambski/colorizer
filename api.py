import cv2 as cv
from google.cloud import storage
import os

def getting_file_names():
    '''
    cette fonction recupere le nom des fichiers dans le bucket
    Requiere le package:
    from google.cloud import storage

    Returns un fichier avec les file names en jpg
    '''
    client = storage.Client()
    bucket = client.bucket("colorizer")

    file_names = []

    for blob in bucket.list_blobs():
        if blob.name.endswith('.jpg'):
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




    # img_bytes = bloby.download_as_bytes()
    # image = Image.open(io.BytesIO(img_bytes))
    # arr = np.array(image)
