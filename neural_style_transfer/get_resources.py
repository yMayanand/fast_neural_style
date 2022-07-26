import os
import zipfile
from urllib import request

def download_and_extract(url, fname):
    request.urlretrieve(url, fname)

    if not os.path.exists(fname.split('.')[0]):
        with zipfile.ZipFile(fname, 'r') as f:
            f.extractall()