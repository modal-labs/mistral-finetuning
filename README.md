# Fine-tuning Mistral 7B on a single GPU with QLoRA

This simple guide will help you fine-tune any language model to make it better at a specific task. With Modal, you can do this training and serve your model in the cloud in minutes - without having to deal with any infrastructure headaches like building images and setting up GPUs.

For this guide, we train [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on a single GPU using [QLoRA](https://github.com/artidoro/qlora), an efficient fine-tuning technique that combines quantization with LoRA to reduce memory usage while preserving task performance. We use 4-bit quantization and train our model on the [SAMsum dataset](https://huggingface.co/datasets/samsum), an existing dataset that summarizes messenger-like conversations in the third person. Modal's easy [GPU-accelerated setup](https://modal.com/docs/guide/gpu) and [built-in storage system](https://modal.com/docs/guide/volumes) help us kick off training in no time.

It's easy to tweak this repository to fit your needs: 
- To train another language model, define `BASE_MODEL` with the desired model name (`common.py`)
- To use your own training data (saved in local .csv or .jsonl files), upload your test and validation datasets to your modal.Volume using the CLI command `modal volume put training-data-vol /local_path/to/dataset /training_data`. Make sure to modify the prompt templates to match your dataset.
- To change the quantization parameters (or do without quantization altogether), modify the `BitsandBytesConfig` in `train.py` (and make sure to apply the same modifications in `inference.py`)

## Before we start - set up a Modal account
1. Create an account on [modal.com](https://modal.com/).
2. Install `modal` in your current Python virtual environment (`pip install modal`)
3. Set up a Modal token in your environment (`python3 -m modal setup`)
4. If you want to monitor your training runs using Weights and Biases, you need to have a [secret](https://modal.com/secrets) named `my-wandb-secret` in your Modal workspace. Only the `WANDB_API_KEY` is needed, which you can get if you log into your Weights and Biases account and go to the [Authorize page](https://wandb.ai/authorize).

## Training
To launch a training job, use:
```
modal run train.py
```

Flags:
- `--detach`: don't terminate app when your local process dies or disconnects (i.e. makes sure you don't accidentally terminate your training job when you close your terminal).
```
modal run --detach train.py
```
- `--resume-from-checkpoint`: resume training from a certain checkpoint saved to your results volume.
```
modal run train.py --resume-from-checkpoint /results/<checkpoint-number>
```

You should make sure your adapter weights have been properly saved in your results volume by running:
```
modal volume ls results-vol
```

## Inference
To try out your freshly fine-tuned model, use:
```
modal run inference.py
```

## Next steps
- Serve your model's inference function using a Modal [web endpoint](https://modal.com/docs/guide/webhooks). Note that leaving an endpoint deployed on Modal doesn't cost you anything, since we scale them serverlessly. See our [QuiLLMan](https://github.com/modal-labs/quillman/) repository for an example of a FastAPI server with an inference endpoint (in `quillman/src/app.py`)
- Try training another model. If the model you'd like to fine-tune is too large for single-GPU training, take a look at our [Llama finetuning repository](https://github.com/modal-labs/llama-finetuning/), which uses FSDP to scale training optimally with multi-GPU.