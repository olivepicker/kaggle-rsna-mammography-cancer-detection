## kaggle RSNA Screening Mammography Breast Cancer Detection, 46th placed solution



</br>

`hyperparameters`

* loss_function : Focal Loss, Delta 4, Alpha 0.5</br>
* optimizer : RAdam </br>
* drop_rate : 0.4 </br>
* weight_decay : 1e-5 </br>
* learning_rate : 3e-5</br>
</br>

`Augmentation`
```python
    train_transform = albumentations.Compose([
            albumentations.LongestMaxSize(max_size=1536,p=1),
            albumentations.PadIfNeeded(min_width = 930, p=1),
            albumentations.ShiftScaleRotate(     
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20, p=0.3),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.2),
            albumentations.OneOf([
                albumentations.GridDistortion(p=0.3),
                albumentations.OpticalDistortion(p=0.3)],p=1.0),          
            albumentations.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit = 0.03, p=0.2),
            albumentations.CLAHE(clip_limit=0.03, tile_grid_size=(8, 8), p=0.2),
            albumentations.Perspective((0.05,0.09), p=0.3),
            albumentations.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, always_apply=False, p=0.3),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1.0),          
            ], p=0.1),  
```

`Highest Submissions`
* 5 Fold of Convnext Large, lb public 0.56, lb private 0.45 
</br>

`Not Worked`
* I've tried so many things, but nothing has succeeded.
* LSTM Model
* Externel Data with Pseudo-Labeling
* Ensemble of 3 different CLAHE
* Multi-modal(Tabnet + CNN..)
</br>

`What I Learned`
* TensorRT ! Really useful, powerful for model inference
* Nvidia Dali, Dicom data processing with GPU

</br>
