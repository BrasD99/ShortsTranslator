import gdown
import os
import zipfile

url = 'https://drive.google.com/uc?id=1dYy24q_67TmVuv_PbChe2t1zpNYJci1J'
output = 'weights.zip'
gdown.download(url, output, quiet=False)

os.mkdir('weights')

with zipfile.ZipFile('weights.zip', 'r') as zip_ref:
    zip_ref.extractall('weights')

os.remove('weights.zip')