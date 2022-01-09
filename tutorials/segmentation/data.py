
import os
import zipfile
import requests


DATA_PATH = './data'
RESULT_PATH = './result'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

print('Downloading and extracting data...')
url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/archive/promise12/datasets-promise12.zip' 
r = requests.get(url,allow_redirects=True)
temp_file = 'temp.zip'
_ = open(temp_file,'wb').write(r.content)

with zipfile.ZipFile(temp_file,'r') as zip_obj:
    zip_obj.extractall(DATA_PATH)
os.remove(temp_file)
print('Done.')
print('Promise12 data downloaded: %s' % os.path.abspath(os.path.join(DATA_PATH,'datasets-promise12')))

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
    print('Result directory created: %s' % os.path.abspath(RESULT_PATH))
