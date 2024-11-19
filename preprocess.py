import numpy as np
import SimpleITK as sitk
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def apply_window(image, window_center, window_width):
        """
        Apply windowing to the image for CT soft tissue visibility.
        """
        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2
        windowed_image = np.clip(image, min_value, max_value)
        windowed_image = ((windowed_image - min_value) / (max_value - min_value)) * 255
        return windowed_image.astype(np.uint8)


def get_image(img_path):
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).squeeze()
        hard_image = apply_window(image, window_center=40, window_width=400)/255
        soft_image = apply_window(image, window_center=-269, window_width=1518)/255
        image = np.stack([hard_image,soft_image],axis=-1)
        return image

def get_mask(mask_path):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).squeeze()
        mask = np.stack([(mask==x).astype(np.float32) for x in [0,1]],-1)
        return mask_path

def get_transform(mode='train'):
        if mode =='train':
                return A.Compose([
                        A.Resize(width=224, height=224),
                        A.HorizontalFlip(),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                        A.ShiftScaleRotate(shift_limit=(-0.05,0.05),scale_limit=(-0.1,0.2),rotate_limit=(-10,10),p=0.5),
                        A.Normalize(mean=0.5,std=0.5,max_pixel_value=1.0, always_apply=True, p=1.0),
                        ToTensorV2(transpose_mask=True)])
        else:
                return A.Compose([
                        A.Resize(width=224, height=224),
                        A.Normalize(mean=0.5,std=0.5,max_pixel_value=1.0, always_apply=True, p=1.0),
                        ToTensorV2(transpose_mask=True)])