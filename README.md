This is offical codes TripleDNet: Exploring Depth Estimation with Self-Supervised Representation Learning

If you find our work useful in your research please consider citing our paper:
```
@inproceedings{triplednetdepth,
  title={TripleDNet: Exploring Depth Estimation with Self-Supervised Representation Learning},
  author={Senturk, Ufuk Umut and Akar, Arif and Ikizler-Cinbis, Nazli},
  booktitle={BMVC},
  year={2022}
}
```

## Models
Coming soon!

## Setup

### Requirements:
- PyTorch1.1+, Python3.5+, Cuda10.0+
- mmcv==0.4.4

```bash
conda create --name featdepth python=3.7
conda activate featdepth

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install pip

# install required packages from requirements.txt
pip install -r requirements.txt
```

## KITTI training data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

## API
We provide an API interface for you to predict depth and pose from an image sequence and visulize some results.
They are stored in folder 'scripts'.

```
eval_depth.py is used to obtain kitti depth evaluation results.
```

```
infer.py or infer_singleimage.py is used to generate depth maps from given models.
```

## Training
You can use following command to launch distributed learning of our model:
```shell
/path/to/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config /path/to/cfg_kitti_tripleD.py --work_dir /dir/for/saving/weights/and/logs'
```

or undistributed learning

```shell
/path/to/python -m --master_port=9900 --config /path/to/cfg_kitti_tripleD.py --work_dir /dir/for/saving/weights/and/logs'
```

Here nproc_per_node refers to GPU number you want to use, which is 4 in our case.

## Acknowledgement

This repository is based on [FeatDepth](https://github.com/sconlyshootery/FeatDepth). 
We thank authors for their contributions. 