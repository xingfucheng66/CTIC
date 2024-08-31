# CTIC: Concept Transfer in Image Classification
## Model Structure
[Structure Figure](fig/CTIC.pdf)
## Abstract
Learning visual concepts has gained significant attention in recent years. However, the concepts learned are often limited to specific datasets and exhibit poor transferability to new tasks. Transfer learning offers a potential solution to this issue, but previous methods frequently fail to improve performance and may even reduce classification accuracy. In this paper, we introduce a novel approach for Concept Transfer in Image Classification (CTIC) that incorporates Low-Rank Adaptation (LoRA) modules, termed CTIC-LoRA. Initially, we pretrain a Bottleneck Concept Learning (BotCL) model using the ImageNet dataset to extract visual concepts. We then integrate CTIC-LoRA modules to augment the BotCL's concept extractor, enabling the model to more accurately identify key components within images and thereby enhancing its transferability across different datasets. Experimental results demonstrate that our method significantly improves both visual performance and classification accuracy, offering new methodologies and insights for future cross-dataset concept transfer research.
## Usage

#### Data Set
Download CUB or ImageNet and set them into direction of your "dataset_dir". You can also make your own dataset with the structure similar to ImageNet and name it as Custom.
