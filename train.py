from modal import Secret
from transformers import TrainerCallback

from common import stub, BASE_MODEL, MODEL_PATH, VOLUME_CONFIG

WANDB_PROJECT = "hf-mistral7b-finetune"

# Callback function to store model checkpoints in modal.Volume
class CheckpointCallback(TrainerCallback):
    def __init__(self, volume):
        self.volume = volume

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print("running commit on modal.Volume after model checkpoint")
            self.volume.commit()
            

# Download training dataset from Hugging Face and push to modal.Volume. 
# You can load in your own dataset to push to a Volume here or push local data files 
# using `modal volume put VOLUME_NAME [LOCAL_PATH] [REMOTE_PATH]` in your CLI.
@stub.function(volumes=VOLUME_CONFIG)
def download_dataset():
    import os
    from datasets import load_dataset

    if not os.path.exists('/training_data/data_train.csv'):
        dataset = load_dataset('gem/viggo')  # downloading data from hugging face
        for split, dataset in dataset.items():
            dataset.to_csv(f"/training_data/data_{split}.csv") # writing data to training data Volume (mounted at /training-data in container)
        
    stub.training_data_volume.commit()


@stub.function(
    gpu="A100",
    secret=Secret.from_name("my-wandb-secret") if WANDB_PROJECT else None,
    timeout=60 * 60 * 2,
    volumes=VOLUME_CONFIG,
)
def finetune(
    model_name: str, 
    wandb_project: str = "", 
    resume_from_checkpoint: str = None  # path to checkpoint in Volume (e.g. "/results/checkpoint-300/")
):
    import os
    from datetime import datetime
    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config)  # Load and quantize the pretrained model baked into our modal.Image
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt):
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
        The order your list the attributes within the function must follow the order listed above. For example the 'name' attribute must always come before the 'exp_release_date' attribute, and so forth.
        For each attribute, fill in the corresponding value of the attribute in brackets. A couple of examples are below. Note: you are to output the string after "Output: ". Do not include "Output: " in your answer.
        
        ### Target sentence:
        {data_point["target"]}

        ### Meaning representation:
        {data_point["meaning_representation"]}
        """

        return tokenize(full_prompt)
    
    # Load datasets from training data Volume
    train_dataset = load_dataset('csv', data_files='/training_data/data_train.csv', split='train')
    eval_dataset = load_dataset('csv', data_files='/training_data/data_validation.csv', split='train')

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if len(wandb_project) > 0:
        # Set environment variables if Weights and Biases is enabled
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_WATCH"] = "gradients"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[CheckpointCallback(stub.results_volume)], # Callback function for committing a checkpoint to Volume when reached
        args=transformers.TrainingArguments(
            output_dir="/results",  # Must also set this to write into results Volume's mount location
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=500,  # Feel free to tweak to correct for under/overfitting
            learning_rate=2.5e-5, # ~10x smaller than Mistral's learning rate
            logging_steps=50,
            bf16=True,
            optim="adamw_8bit",
            logging_dir="/results/logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=50,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=50,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb" if WANDB_PROJECT else "",         
            run_name=f"mistral7b-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  if WANDB_PROJECT else ""     
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    model.config.use_cache = False  # Silence the warnings. Re-enable for inference!
    trainer.train()  # Run training
    
    model.save_pretrained(f"/results")  # Store fully trained model in results Volume
    stub.results_volume.commit()


@stub.local_entrypoint()
def main(resume_from_checkpoint: str = None):
    print("Downloading data from Hugging Face and syncing to volume.")
    download_dataset.remote()
    print("Finished syncing data.")

    print("Starting training run.")
    finetune.remote(model_name=BASE_MODEL, wandb_project=WANDB_PROJECT, resume_from_checkpoint=resume_from_checkpoint)
    print("Completed training!")
    print("To test your trained model, run `modal run inference.py`")