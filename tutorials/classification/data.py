
import os
import requests


DATA_PATH = './data'
RESULT_PATH = './result'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

print('Downloading and extracting data...')
url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/raw/fetal/ultrasound_50frames.h5' 
r = requests.get(url,allow_redirects=True)
filename = os.path.join(DATA_PATH,'ultrasound_50frames.h5')
_ = open(filename,'wb').write(r.content)
print('Done.')
print('Image and label data downloaded: %s' % filename)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
    print('Result directory created: %s' % os.path.abspath(RESULT_PATH))