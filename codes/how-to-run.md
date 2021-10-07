# Laplacian Pyramid Networks: A new Approach for Multispectral Pansharpening

Homepage: 

https://chengjin.netlify.app/

https://liangjiandeng.github.io/

- Code for paper: "Laplacian Pyramid Networks: A new Approach for Multispectral Pansharpening, Information Fusion"
- State-of-the-art pansharpening performance

This is the description of how to run our training code and testing code. 

## Training instructions

In `training.py`:

- Modify the train/validation data input in .mat format
- Change the `image_size` to the dataset input
- Change the MTF kernel input according to the dataset settings in the comments
- Change the normalizaition scale according to the worldview level of the corresponding datasets

In `model.py`:
- Modify `num_feature`, `num_ms_channels`
- Change output channel number of image alignment step
- Change the MTF kernel input according to the dataset settings in the comments

## Testing instructions

In `model.py`:
- Modify `num_feature`, `num_ms_channels` according to the dataset settings in the comments
- Change output channel number of image alignment step according to the dataset settings in the comments
- Change the MTF kernel input according to the dataset settings in the comments

In `testing.py`:
- Load (pretrained) model path
- Load test data
- Change the MTF kernel input according to the dataset settings in the comments

## Third-Party datasets stucture
LPPN supports datasets other than WorldView-3, QuickBird and GaoFen-2. If you want to test on your own dataset, make sure to convert your dataset into `.mat` format and contains the following sturcture:

```
YourDataset.mat
|--ms: original multispectral images in .mat format, basically have the size of N*h*w*C 
|--lms: interpolated multispectral images in .mat format, basically have the size of N*H*W*C 
|--pan: original panchromatic images in .mat format, basically have the size of N*H*W*1
|--gt: simulated ground truth images images in .mat format, basically have the size of N*H*W*C 
```
