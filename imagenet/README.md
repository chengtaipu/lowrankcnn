# ImageNet Experiments

## Installation

In this repository, we use a self-brewed caffe, which provides Batch Normalization and faster MPI-based parallel training. In case you don't need such features, you may replace it with your own caffe.

The detailed installation instructions of the self-brewed caffe can be found [here](https://github.com/yjxiong/caffe).

## Make a low-rank net

You can use `lowrank_approx.py` to make a low-rank net. For example,

    python2 lowrank_approx.py \
      --model models_vgg/vgg_deploy.prototxt \
      --config models_vgg/config.json \
      --save_model models_vgg/vgg_lowrank_deploy.prototxt

This will make a low-rank VGGNet prototxt only.

You can also download some pretrained models

+  [CaffeNet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
+  [VGGNet](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)

And use the tool to convert them to low-rank ones

    python2 lowrank_approx.py \
      --model models_vgg/vgg_deploy.prototxt \
      --config models_vgg/config.json \
      --save_model models_vgg/vgg_lowrank_deploy.prototxt \
      --weights VGG_ILSVRC_16_layers.caffemodel \
      --save_weights vgg_lowrank.caffemodel

After finetuning the lowrank model for about one epoch, its accuracy will be similar to the original model.

## Results

The actual speedup could depend on the hardware and software implementation. Below are the results tested on a Titan Black with cuda-7.5 and cudnn-v4. The batch size is set to 256 for CaffeNet, and 32 for VGG16. The Average Forward-Backward time per minibatch is reported.

|                  | Top-5 Acc. (%) | Time (ms) / iter | Actual speedup |
|------------------|---------------:|-----------------:|---------------:|
| CaffeNet         |          80.03 |              668 |              - |
| CaffeNet-Lowrank |          79.66 |              307 |          2.18× |
| VGG16            |          90.60 |             1570 |              - |
| VGG16-Lowrank    |          90.31 |              759 |          2.07× |
