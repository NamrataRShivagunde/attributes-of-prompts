from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

checkpoint = "facebook/opt-30b"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, "facebook/opt-30b", device_map="auto", no_split_module_classes=["GPTJBlock"]
)

print(model.hf_device_map)