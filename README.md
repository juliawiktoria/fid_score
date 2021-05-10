# FID score wrapper

This is a small wrapper for calculating Fréchet inception distance (FID) between a directory with (generated) images and a FID statistics file for the CIFAR10 dataset. It is a small part of my undergraduate thesis.

The wrapper uses [kklemon's in-code reimplementation](https://github.com/kklemon/pytorch-fid) of the [mseitzer's package](https://github.com/mseitzer/pytorch-fid).