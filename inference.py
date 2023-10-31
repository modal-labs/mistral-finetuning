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
            "/results/",  # Get finetuned model from results Volume (mounted at "/results")
            device_map="auto",
            trust_remote_code=True,
        )

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            add_bos_token=True,
            trust_remote_code=True,
        )

    # Insert user input into prompt template
    def generate_prompt(self, target_sentence: str = ""):
        eval_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {target_sentence}

        ### Meaning representation:
        """

        return self.eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # Inference function with base pretrained model
    @method()
    async def generate_base(self, target_sentence: str):
        import torch

        model_input = self.generate_prompt(target_sentence)

        self.pretrained_model.eval()
        with torch.no_grad():
            print(self.eval_tokenizer.decode(self.pretrained_model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True))

    # Inference function with finetuned model
    @method()
    async def generate_ft(self, target_sentence: str):
        import torch

        model_input = self.generate_prompt(target_sentence)

        self.finetuned_model.eval()
        with torch.no_grad():
            print(self.eval_tokenizer.decode(self.finetuned_model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True))


@stub.local_entrypoint()
def main():
    sentence = "One thing I thought that makes Far Cry 3 a pretty good game is that it has multiplayer as well. It has a great single-player campaign, but you can jump online and enjoy it that way too."
    model = Model()
    model.generate_ft.remote(sentence)
