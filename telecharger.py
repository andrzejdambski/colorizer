def telecharger_url_qui_contient_tensor(url):
    '''
    Fonction qui telecharge le fichier dans le URL
    necessite - import requests, from io import BytesIO, import tensorflow

    Resort la data dans le lien, soit un tensor dans ce cas
    '''
    # url = "https://storage.googleapis.com/colorizer/L_normalized.npy"
    data = requests.get(url).content
    array = np.load(BytesIO(data))
    tensor = tf.convert_to_tensor(array)
    return tensor
