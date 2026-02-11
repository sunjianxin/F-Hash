## F-Hash: *Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding*

![results](assets/animation_flame.gif)
Demo video can be found <a href="https://youtu.be/AiN_mFc_Oig?si=8QNchPEweSy_SrxO" target="_blank">here</a>.

### 1. Agent
Agentic AI controlling the workflows

### 2. Visualization Tools
Visualization tools for agent to call as needed

####  Dependencies
- C++11 or higher compiler.
- [VTK](https://github.com/Kitware/VTK), The Visualization Toolkit (VTK).
- [MPI](http://www.mpich.org)

#### Build
1. Install VTK
    * Follow the [instruction](https://docs.vtk.org/en/latest/build_instructions/index.html) to build VTK
    * Another use full resource can be found [here](https://www.cs.purdue.edu/homes/xmt/classes/CS530/spring2018/project0.html)
2. TODO: Get and build example code
```bash
git clone https://github.com/sunjianxin/vtk
cd vtk
mkdir build
cd build
cmake ..  \
-DCMAKE_CXX_COMPILER=mpicxx \
-DVTK_DIR:PATH=path_to_VTK_DVR-MFA_installation_folder
```
*path_to_mfa_include_folder* is the folder location in the project folder in step 2. *path_to_VTK_DVR-MFA_installation_folder* is the installation location when you configure VTK_DVR-MFA before building, and it is normally at */usr/local/include/vtk-version* by default.
