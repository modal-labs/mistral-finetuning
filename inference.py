from modal import method
from common import stub, MODEL_PATH, VOLUME_CONFIG
    

@stub.cls(gpu="A100", volumes=VOLUME_CONFIG)
class Model:
    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

       # Quantization config, make sure it is the same as the one used during training 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True,
        )

        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            "/results/",  # Get finetuned model from results Volume (mounted at "/results" in our container)
            device_map="auto",
            trust_remote_code=True,
        )

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            add_bos_token=True,
            trust_remote_code=True,
        )


    # Insert user input into prompt template
    def generate_prompt(self, message: str = ""):
        eval_prompt = f"[INST] <<SYS>>\nUse the Input to provide a summary of a conversation.\n<</SYS>>\n\nInput:\n{message} [/INST]\n\nSummary:"

        return self.eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # Inference function with base pretrained model
    @method()
    async def generate_base(self, target_sentence: str):
        import torch

        model_input = self.generate_prompt(target_sentence)

        self.pretrained_model.eval()
        with torch.no_grad():
            print(self.eval_tokenizer.decode(self.pretrained_model.generate(**model_input, max_new_tokens=100, eos_token_id=self.eval_tokenizer.eos_token_id)[0], skip_special_tokens=True))

    # Inference function with finetuned model
    @method()
    async def generate_ft(self, target_sentence: str):
        import torch

        model_input = self.generate_prompt(target_sentence)

        self.finetuned_model.eval()
        with torch.no_grad():
            print(self.eval_tokenizer.decode(self.finetuned_model.generate(**model_input, max_new_tokens=100, eos_token_id=self.eval_tokenizer.eos_token_id)[0], skip_special_tokens=True))


@stub.local_entrypoint()
def main():
    message = "Eric: MACHINE! Rob: That's so gr8! Eric: I know! And shows how Americans see Russian ;) Rob: And it's really funny! Eric: I know! I especially like the train part! Rob: Hahaha! No one talks to the machine like that! Eric: Is this his only stand-up? Rob: Idk. I'll check. Eric: Sure. Rob: Turns out no! There are some of his stand-ups on youtube. Eric: Gr8! I'll watch them now! Rob: Me too! Eric: MACHINE! Rob: MACHINE! Eric: TTYL? Rob: Sure :)"
    
    print(Model().generate_base.remote(message))
    print(Model().generate_ft.remote(message))
