## F-Hash: *Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding*

![results](assets/animation_flame.gif)

F-Hash is a novel feature-based multi-resolution Tesseract encoding architecture to greatly enhance the convergence speed compared with existing input encoding methods for modeling time-varying volumetric data. The proposed design incorporates multi-level collision-free hash functions that map dynamic 4D multi-resolution embedding grids without bucket waste, achieving high encoding capacity with compact encoding parameters. Our encoding method is agnostic to time-varying feature detection methods, making it a unified encoding solution for feature tracking and evolution visualization.

[Github Page](https://github.com/sunjianxin/F-Hash), 
[ArXiv](https://arxiv.org/abs/2507.03836),
[Publishers' Version](https://ieeexplore.ieee.org/abstract/document/11261881)

Demo video can be found <a href="https://youtu.be/AiN_mFc_Oig?si=8QNchPEweSy_SrxO" target="_blank">here</a>.

### 1. Packages
```bash
pip install vtk torch tqdm 
```
### 2. Directory tree
```
F-Hash (Source code)
├── data
│   └── argon_bubble  (Raw dataset: .dat)
│       ├── argon_128x128x256  (Keyframe: .vtk and .bin)
│       │   └── feature_local (Training data: .bin; Meta data: .txt, .json)
│       └── argon_128x128x256_predict
│           └── f_hash
│               └── time_step_index (Prediction of time step during training: .vtk)
└── models
    └── argon_128x128x256
        └── f_hash (saved checkpoints during training)
```
### 2. Run
Coreset Selection
```bash
python coreset.py
```
Training
```bash
python train.py
```
Testing
```bash
python test.py
```
### TODO
- [X] Coreset selection
- [X] F-Hash input encoding
- [x] Training
- [ ] Adaptive Ray Marching (ARM)

## Citing H-Hash
If you use it in your research, we would appreciate a citation via
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
  doi={10.1109/TVCG.2025.3634812}
}
```

## License
F-Hash is distributed under the terms of the BSD-3 license.