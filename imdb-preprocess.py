
# coding: utf-8

# # IMDB-WIKI
# ##  Multi-task age and gender classification

# On the original paper [DEX: Deep EXpectation of apparent age from a single image](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf) the authors were able to display remarkable results in classifying the age of an individual based on a given image alone. 
# 
# Let see how accuracy (bad I guess), with limited resources, we can get with self-construct architecture. And not only age, we also classifying gender by using multi-task training technique.

# In[1]:


import os
from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import scipy.misc as spm
from scipy import ndimage
import datetime
import matplotlib.image as plt
from IPython.display import Image, display
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from collections import Counter
# from skimage.transform import resize

IMG_DIR = os.path.expanduser("~")+'/coding/cnn/datasets/imdb_crop'
MAT_FILE = os.path.expanduser("~")+'/coding/cnn/datasets/imdb_crop/imdb.mat'


def reformat_date(mat_date):
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


# In[3]:


def create_path(path):
    return os.path.join(IMG_DIR, path[0])


# In[4]:


mat_struct = sio.loadmat(MAT_FILE)
data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

keys = ['dob',
    'photo_taken',
    'full_path',
    'gender',
    'name',
    'face_location',
    'face_score',
    'second_face_score',
    'celeb_names',
    'celeb_id'
]

imdb_dict = dict(zip(keys, np.asarray(data_set)))
imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

# Add 'age' key to the dictionary
imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

print("Dictionary created...")


# The IMDB dataset has total 460,723 face images from 20,284 celebrities. 
# 
# We will ignore:
# * images with more than one face
# * gender is NaN
# * invalid age.
# 
# As we are using only a subset of the data, and also using a self-constructed model that has a much smaller capacity, thus we need to take steps to adjust accordingly.
# 
# ~~The original paper uses 101 age classes, which was appropriate for the their data set size and learning architecture. As we are only using a small subset of the data and a very simple model, the number of classes was set to 4:~~
# * Young    (age < 30yrs)
# * Middle   (30 <= age <45)
# * Old      (45 <= age < 60)
# * Very Old (60 <= age)
# 
# Another approach, 101 classes, age label from 0..100

# In[ ]:


for _ in range(5):
    np.random.shuffle(X_age)
    np.random.shuffle(X_gender)


# In[37]:


raw_path = imdb_dict['full_path']
raw_age = imdb_dict['age']
raw_gender = imdb_dict['gender']
raw_sface = imdb_dict['second_face_score']

age = []
gender = []
imgs = []
current_age = np.zeros(101)
for i, sface in enumerate(raw_sface):
    if np.isnan(sface) and raw_age[i] >= 0 and raw_age[i] <= 100 and not np.isnan(raw_gender[i]):
        age_tmp = 0;
        if current_age[raw_age[i]] >= 5000:
            continue
        age.append(raw_age[i])
        gender.append(raw_gender[i])
        imgs.append(raw_path[i])
        current_age[raw_age[i]] += 1


# In[38]:


sns.distplot(age);
print("Age size: " + str(len(age)))


# In[39]:


counter = Counter(age)
print(counter)


# In[40]:


pickle_file = 'imdb-gender-age101.pkl'

try:
    f = open(os.getcwd()+"/pkl_folder/"+pickle_file, 'wb')
    save = {
    'age': age,
    'gender': gender,
    'imgs': imgs
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[9]:


raw_path = imdb_dict['full_path']
raw_age = imdb_dict['age']
raw_gender = imdb_dict['gender']
raw_name = imdb_dict['name']
raw_face_location = imdb_dict['face_location']
raw_face_score = imdb_dict['face_score']
raw_second_face_score = imdb_dict['second_face_score']
raw_celeb_names = imdb_dict['celeb_names']
raw_celeb_id = imdb_dict['celeb_id']


# In[1]:


for i in range(100):
    display(Image(filename=raw_path[i]))
    print("Path: " + str(raw_path[i]))
    print("Age: " + str(raw_age[i]))
    print("Gender: " + str(raw_gender[i]))
    print("Name: " + str(raw_name[i]))
    print("Face location: " + str(raw_face_location[i]))
    print("Face score: " + str(raw_face_score[i]))
    print("Second face score: " + str(raw_second_face_score[i]))
    print("Celeb id: " + str(raw_celeb_id[i])+"\n")
    
    

