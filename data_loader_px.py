import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

class DataLoader():
    def __init__(self, datadir, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.list_a = sorted(glob(datadir+'/%s/img*' % dataset_name))
        self.list_b = sorted(glob(datadir+'/%s/hed*' % dataset_name))
        self.num=len(self.list_a)
        #print(len(self.list_a),len(self.list_b))

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
#        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_idx=np.random.randint(0,len(self.list_a),batch_size)
#        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_idx in batch_idx:
            img_A = self.imread2(self.list_a[img_idx], self.img_res)
            img_B = self.imread2(self.list_b[img_idx], self.img_res)

            #img_A = self.imresize(img_A, self.img_res)
            #img_B = self.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"

        self.n_batches = int(len(self.list_a) / batch_size)
        
        for i in range(self.n_batches-1):
            batch_idx=np.random.randint(0,len(self.list_a),batch_size)
            imgs_A, imgs_B = [], []
            for img_idx in batch_idx:
                img_A = self.imread2(self.list_a[img_idx], self.img_res)
                img_B = self.imread2(self.list_b[img_idx], self.img_res)
                #img_A = self.imresize(img_A, self.img_res)
                #img_B = self.imresize(img_B, self.img_res)
               

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            yield imgs_A, imgs_B

    def imread(self, path):
        return imageio.imread(path, mode='RGB').astype(np.float)
    
    def resize(self, img, sz):
        return np.asarray(Image.fromarray(img).resize(sz, resample=2))
        
    def imread2(self, img, sz):
        return np.asarray(Image.open(img).convert("RGB").resize(sz, resample=2), dtype=np.float)    
