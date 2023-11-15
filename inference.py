from modal import method
from common import stub, MODEL_PATH, VOLUME_CONFIG
    

@stub.cls(gpu="A100", volumes=VOLUME_CONFIG)
class Model:
    def __init__(self, run_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

       # Quantization config, make sure it is the same as the one used during training 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True,
        )

        if run_id:
            self.model = PeftModel.from_pretrained(  # model with adapter
                base_model,
                f"/results/{run_id}",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = base_model

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            add_bos_token=True,
            trust_remote_code=True,
            padding_side="right",
        )

        self.template = "[INST] <<SYS>>\nUse the Input to provide a summary of a conversation.\n<</SYS>>\n\nInput:\n{message} [/INST]\n\nSummary:"

    def tokenize_prompt(self, prompt: str = ""):
        return self.eval_tokenizer(prompt, return_tensors="pt").to("cuda")

    @method()
    async def generate(self, message: str):
        import torch

        model_input = self.tokenize_prompt(self.template.format(message=message))

        self.model.eval()
        with torch.no_grad():
            print(self.eval_tokenizer.decode(self.model.generate(**model_input, max_new_tokens=100, eos_token_id=self.eval_tokenizer.eos_token_id)[0], skip_special_tokens=True))


@stub.local_entrypoint()
def main(run_id: str):
    if not run_id:
        print("Warning: run_id not found. Please input run_id from previous training run to generate with trained adapter.")
        print("Usage with trained adapter: modal run inference.py --run_id <run_id>")

    messages = [
        "Eric: MACHINE! Rob: That's so gr8! Eric: I know! And shows how Americans see Russian ;) Rob: And it's really funny! Eric: I know! I especially like the train part! Rob: Hahaha! No one talks to the machine like that! Eric: Is this his only stand-up? Rob: Idk. I'll check. Eric: Sure. Rob: Turns out no! There are some of his stand-ups on youtube. Eric: Gr8! I'll watch them now! Rob: Me too! Eric: MACHINE! Rob: MACHINE! Eric: TTYL? Rob: Sure :)", 
        "Ollie: Hi , are you in Warsaw Jane: yes, just back! Btw are you free for diner the 19th? Ollie: nope! Jane: and the 18th? Ollie: nope, we have this party and you must be there, remember? Jane: oh right! i lost my calendar.. thanks for reminding me Ollie: we have lunch this week? Jane: with pleasure! Ollie: friday? Jane: ok Jane: what do you mean 'we don't have any more whisky!' lol.. Ollie: what!!! Jane: you just call me and the all thing i heard was that sentence about whisky... what's wrong with you? Ollie: oh oh... very strange! i have to be carefull may be there is some spy in my mobile! lol Jane: dont' worry, we'll check on friday. Ollie: don't forget to bring some sun with you Jane: I can't wait to be in Morocco.. Ollie: enjoy and see you friday Jane: sorry Ollie, i'm very busy, i won't have time for lunch tomorrow, but may be at 6pm after my courses?this trip to Morocco was so nice, but time consuming! Ollie: ok for tea! Jane: I'm on my way.. Ollie: tea is ready, did you bring the pastries? Jane: I already ate them all... see you in a minute Ollie: ok", 
        "Rita: I'm so bloody tired. Falling asleep at work. :-( Tina: I know what you mean. Tina: I keep on nodding off at my keyboard hoping that the boss doesn't notice.. Rita: The time just keeps on dragging on and on and on.... Rita: I keep on looking at the clock and there's still 4 hours of this drudgery to go. Tina: Times like these I really hate my work. Rita: I'm really not cut out for this level of boredom. Tina: Neither am I."
    ]

    print("=" * 20 + "Generating without adapter" + "=" * 20)
    for summary in Model().generate.map(messages):
        print(summary)

    if run_id:
        print("=" * 20 + "Generating with adapter" + "=" * 20)
        for summary in Model(run_id=run_id).generate.map(messages):
            print(summary)
