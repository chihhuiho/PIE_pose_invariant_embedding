# PIE_pose_invariant_embedding
Implementation of the research paper [PIEs: Pose Invariant Embeddings (CVPR2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ho_PIEs_Pose_Invariant_Embeddings_CVPR_2019_paper.pdf) 
The result presented in the paper is averaged over 5 rounds, so it might be slightly different from the model provided.


## Usage

1. Install all required library
```
conda env create -f environment.yml --name <env_name>
```

2. Download ModelNet40 dataset 
```
download_modelnet40.sh
```

3. Run cnn based methods (cnn, mvcnn, picnn) by: 
```
cd cnn_based
cd <cnn, mvcnn, picnn>
sh run.sh
```

4. Run proxy based methods (proxy, mvproxy, piproxy) by: 
```
cd proxy_based
cd <proxy, mvproxy, piproxy>
sh run.sh
```

5. Run triplet center based methods (triplet, mvtriplet, pitriplet) by: 
```
cd triplet_center_based
cd <triplet, mvtriplet, pitriplet>
sh run.sh
```

6. Check the result of pretrained models in log folder and log_robustness folder. log_robustness folder shows the classification results from single view to all the views provided. Read run.sh for more information.


## Classification result for pretrained models
The proposed method is similar to single view based methods when only single view is given and similar to multiview based methods (mv___) when all views are given. The proposed method is more robusted to the number of view given during inference time.

### MVCNN
|  |<td colspan=2> Number of views               |
|  | Single view | All views (12) |
:---------------------|:----:|:----:|
| cnn                 | 84.66 | 87.50 | 
| mvcnn               | 77.75 | 89.75 | 
| picnn               | 85.70 | 89.25 |
| proxy               | 85.60 | 88.62 | 
| mvproxy             | 78.39 | 90.38 | 
| piproxy             | 85.49 | 89.25 |
| triplet             | 85.23 | 88.75 | 
| mvtriplet           | 76.94 | 89.38 | 
| pitriplet           | 83.50 | 89.38 |

### ObjectPI

| Number of views | Single view | All views (8) |
:---------------------|:----:|:----:|
| cnn                 | 65.82 | 76.53 | 
| mvcnn               | 59.44 | 77.55 | 
| picnn               | 67.60 | 79.59 |
:---------------------|:----:|:----:|
| proxy               | 69.52  | 79.59 | 
| mvproxy             | 64.03  | 76.53 | 
| piproxy             | 68.62  | 79.59 |
:---------------------|:----:|:----:|
| triplet             | 70.79  | 77.55 | 
| mvtriplet           | 63.65  | 78.57 | 
| pitriplet           | 69.64  | 75.51 |

## Citation
If you mentioned the method in your research, please cite this article:
```
@InProceedings{Ho_2019_CVPR,
author = {Ho, Chih-Hui and Morgado, Pedro and Persekian, Amir and Vasconcelos, Nuno},
title = {PIEs: Pose Invariant Embeddings},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
