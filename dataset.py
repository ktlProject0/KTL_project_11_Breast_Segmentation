import os
import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import SimpleITK as sitk
from preprocess import get_image,get_mask,get_transform
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,direc,mode='eval'):
        self.mode= mode
        img_path = natsorted(glob.glob(os.path.join(direc,'images','*')))
        mask_path = natsorted(glob.glob(os.path.join(direc,'masks','*')))
        img_df = pd.DataFrame({"image_path":img_path})
        img_df['id'] = img_df['image_path'].apply(lambda x:x.split("/")[-1].split(".")[0])
        mask_df = pd.DataFrame({"mask_path":mask_path})
        mask_df['id'] = mask_df['mask_path'].apply(lambda x:x.split("/")[-1].split(".")[0].replace("_mask",""))
        self.meta_df = pd.merge(img_df,mask_df,on='id')

        self.transform =get_transform(self.mode)
        self.cache={}
        
    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self,idx):
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            sample = self.meta_df.iloc[idx,:].to_dict()
            image = get_image(sample['image_path'])
            mask = get_mask(sample['mask_path'])
            
            sample['image'] = image
            sample['mask'] = mask
            sample['origin_shape'] = image.shape
            
            self.cache[idx] = sample
            
        if self.transform:
            transformed = self.transform(image= sample['image'], mask = sample['mask'])

        sample_input = {}
        sample_input['input'] = transformed['image']
        sample_input['target'] = transformed['mask']
        sample_input['origin_shape'] = sample['origin_shape']
        
        return sample_input
        
if __name__ == '__main__':
    train = CustomDataset(direc='./data/train',mode='train')
    test = CustomDataset(direc='./data/test',mode='test')
    
    for sample_input in train:
        print(sample_input['input'].shape,sample_input['target'].shape)


    
        