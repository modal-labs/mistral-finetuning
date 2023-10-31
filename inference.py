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
        The order your list the attributes within the function must follow the order listed above. For example the 'name' attribute must always come before the 'exp_release_date' attribute, and so forth.
        For each attribute, fill in the corresponding value of the attribute in brackets. A couple of examples are below. Note: you are to output the string after "Output: ". Do not include "Output: " in your answer.
        
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
    sentences = [
        "Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?",
        # "One thing I thought that makes Far Cry 3 a pretty good game is that it has multiplayer as well. It has a great single-player campaign, but you can jump online and enjoy it that way too.",
        # "Super Bomberman is an action-strategy game that has received average ratings and can be played on Nintendo and PC, though it's not available on Steam and does not have a Linux or Mac Release.",	
        # "You mean Tony Hawk's Pro Skater 3, the 2001 sports game?",
        # "Horizon: Zero Dawn is an action-adventure, role-playing, shooter with third person player perspective and no multiplayer mode, rated T (for Teen) and released in 2017 by Guerrilla Games.",
        # "What is it about the driving/racing simulators made by Slightly Mad Studios that you find mediocre?",
        # "Stronghold 2 was released in 2005 as a real-time strategy simulation played from the standard bird view perspective. It received an average rating from players.",
        # "Naughty Dog did an amazing job with The Last of Us, and they really made the most of that M rating.",
    ]
    
    for output in Model().generate_ft.map(sentences):
        print(output)
