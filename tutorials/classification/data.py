
import os
import shutil
import requests
import zipfile

DATA_PATH = './data'
RESULT_PATH = './result'

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

print('Downloading and extracting data...')
url = 'https://github.com/YipengHu/datasets/raw/refs/heads/fetal/fetal.zip' 
r = requests.get(url,allow_redirects=True)
temp_file = 'temp.zip'
_ = open(temp_file,'wb').write(r.content)

with zipfile.ZipFile(temp_file, 'r') as zip_ref:
    zip_ref.extractall(DATA_PATH)
os.remove(temp_file)
print('Done.')

filename = os.path.join(DATA_PATH,'fetal.h5')
print('Image and label data downloaded: %s' % filename)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
    print('Result directory created: %s' % os.path.abspath(RESULT_PATH))
    