# F-Hash: *Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding*

F-Hash is a novel feature-based multi-resolution Tesseract encoding architecture to greatly enhance the convergence speed compared with existing input encoding methods for modeling time-varying volumetric data. The proposed design incorporates multi-level collision-free hash functions that map dynamic 4D multi-resolution embedding grids without bucket waste, achieving high encoding capacity with compact encoding parameters. Our encoding method is agnostic to time-varying feature detection methods, making it a unified encoding solution for feature tracking and evolution visualization.

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
Install from a local clone of tiny-cuda-nn, invoke to install Tinycudann to your virtual enviroment
```bash
tiny-cuda-nn$ cd bindings/torch
tiny-cuda-nn/bindings/torch$ python setup.py install
```
### 3. Visualization Tools
Run Coreset Selection
```bash
python coreset.py
```


## Citing H-Hash
```bibtex
@ARTICLE{sun2025fhash,
  author={Sun, Jianxin and Lenz, David and Yu, Hongfeng and Peterka, Tom},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={F-Hash: Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding}, 
  year={2026},
  volume={32},
  number={1},
  pages={396-406},
  keywords={Encoding;Data visualization;Training;Convergence;Rendering (computer graphics);Data models;Superresolution;Hash functions;Computational modeling;Neural radiance field;Time-varying volume;volume visualization;input encoding;deep learning},
  doi={10.1109/TVCG.2025.3634812}}
```