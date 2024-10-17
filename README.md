# Creating Occluded Images: A Self-Occlusion Benchmark for Crowd Pose Estimation
# Dataset
The mask and ours dataset can be found in the [MASK](https://pan.baidu.com/s/1gUI77VdSAC-L8ssSkPmc4Q )
# Usage
We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.   
···  
git clone https://github.com/open-mmlab/mmcv.git  
cd mmcv  
git checkout v1.3.9  
MMCV_WITH_OPS=1 pip install -e .  
cd ..  
git clone https://github.com/123qwaz/aug.git  
cd aug/main  
pip install -v -e .
···  
