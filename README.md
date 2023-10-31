# Fine-tuning Mistral 7B on a single GPU with QLoRA

This simple guide will help you fine-tune any language model to make them better at a specific task. Using Modal, you can train and serve your model in the cloud in minutes - without having to deal with any infrastructure headaches like building images and setting up GPUs.

For this guide, we train [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on a single GPU using [QLoRA](https://github.com/artidoro/qlora), an efficient fine-tuning technique that combines quantization with LoRA to reduce memory usage while preserving task performance. We train our model on an existing dialogue summary dataset, so that our model gets better at producing concise summaries of conversations. We take advantage of Modal's easy GPU-accelerated environment setup and built-in storage system to reduce the amount of time we spend on our training job.

It's easy to tweak this repository to fit your needs: 
- To train another language model, define `BASE_MODEL` with the desired model name (`common.py`)
- To use your own dataset, XX
- To change the quantization parameters (or do without quantization altogether), modify the `bitsandbytes` config in `train.py` (and make sure to apply the same modifications in `inference.py`)

## Before we start - set up a Modal account
1. Create an account on [modal.com](https://modal.com/).
2. Install `modal` in your current Python virtual environment (`pip install modal`)
3. Set up a Modal token in your environment (`modal token new`)
4. If you want to track your training runs using Weights and Biases, you need to have a [secret](https://modal.com/secrets) named `my-wandb-secret` in your workspace. Only the `WANDB_API_KEY` is needed which you can get if you log into Weights and Biases and go to the [Authorize page](https://wandb.ai/authorize).

## Training
To kick off a training job, use:
```
modal run train.py
```

## Inference
To try out your freshly fine-tuned model, use:
```
modal run inference.py
```

If the model you'd like to fine-tune is too large for single-GPU training (i.e. you're running into out-of-memory errors), take a look at our [Llama finetuning repository](https://github.com/modal-labs/llama-finetuning/), which uses FSDP to scale training optimally with multi-GPU.