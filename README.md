# Creating Occluded Images: A Self-Occlusion Benchmark for Crowd Pose Estimation
# Dataset
The mask and ours dataset can be found in the [MASK](https://pan.baidu.com/s/1gUI77VdSAC-L8ssSkPmc4Q )
# Usage
We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.   
```
git clone https://github.com/open-mmlab/mmcv.git  
cd mmcv  
git checkout v1.3.9  
MMCV_WITH_OPS=1 pip install -e .  
cd ..  
git clone https://github.com/123qwaz/aug.git  
cd aug/main  
pip install -v -e .  
```
After downloading the pretrained models, please conduct the experiments by running  
```  
# for single machine
bash main/tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch --seed 0
```  
To test the pretrained models performance, please run  
```  
bash tools/dist_test.sh <Config PATH> <Checkpoint PATH> <NUM GPUs>
```  
# Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection) and [Vitpose](https://github.com/ViTAE-Transformer/ViTPose)  
# Citing
```  
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
@inproceedings{
  xu2024aug,
  title={Creating Occluded Images: A Self-Occlusion dataset for Crowd Pose Estimation},
  author={Lingling Li and Chunxiao Song and Song Wang and Gangtao Han and Enqing Chen and Guanghui Wang},
  year={2024},
}
```  
