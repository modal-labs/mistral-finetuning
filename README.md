# Fine-tuning Mistral 7B using QLoRA

This simple guide will help you fine-tune any language model to make them better at a specific task. By using Modal, you can fine-tune and run your model in the cloud in minutes - without having to deal with any infrastructure headaches like building images and setting up GPUs.

For this guide, we train Mistral 7B on an existing Meaning Representation dataset using QLoRA, an efficient fine-tuning technique that combines quantization with LoRA to reduce memory usage while preserving task performance. We take advantage of Modal's easy GPU-accelerated environment setup and built-in storage system to reduce the amount of time we spend waiting to kick off our training job and ultimately run our trained model. 

It's easy to tweak this repository to fit your needs: 
- To train another language model, define `BASE_MODEL` with the desired model name (`common.py`)
- To use your own dataset,
- To change the quantization parameters (or do without it altogether), modify the `bitsandbytes` config in `train.py` (and make sure to make the same config modifications in `inference.py`)

## Training
To kick off a training job, use:
```
modal run train.py
```

## Inference
