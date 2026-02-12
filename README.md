## F-Hash: *Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding*

![results](assets/animation_flame.gif)
Demo video can be found <a href="https://youtu.be/AiN_mFc_Oig?si=8QNchPEweSy_SrxO" target="_blank">here</a>.

### 1. Packages
```bash
pip install vtk
```
### 2. Install Tinycudann
Get the Tinycudann source from [here](https://github.com/NVlabs/tiny-cuda-nn). Default Tinycudann supporting half-precision. To support Float 32 go into include/tiny-cuda-nn/common.h and change
```bash
#define TCNN_HALF_PRECISION (!(TCNN_MIN_GPU_ARCH == 61 || TCNN_MIN_GPU_ARCH <= 52))
```
to
```bash
#define TCNN_HALF_PRECISION 0
```
Install from a local clone of tiny-cuda-nn, invoke
```bash
tiny-cuda-nn$ cd bindings/torch
tiny-cuda-nn/bindings/torch$ python setup.py install
```
### 3. Visualization Tools
Run Coreset Selection
```bash
python coreset.py
```
