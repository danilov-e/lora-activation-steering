"""Load Qwen2 models and generate one reproducible answer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def load_model_and_tokenizer(settings):
    device = settings["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = settings["dtype"]
    if dtype == "auto":
        dtype = "float16" if device == "cuda" else "float32"
    checkpoint = {"revision": settings["revision"], "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(settings["id"], **checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        settings["id"], **checkpoint, torch_dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_answer(model, tokenizer, question, settings, seed):
    prompt = format_prompt(tokenizer, question)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    set_seed(seed)
    output = model.generate(
        **inputs, **settings, pad_token_id=tokenizer.pad_token_id,
        num_beams=1, num_return_sequences=1,
    )
    tokens = output[0, inputs.input_ids.shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
    if end is not None:
        tokens = tokens[:end + 1]
    return {
        "answer": tokenizer.decode(tokens, skip_special_tokens=True),
        "generated_tokens": len(tokens),
        "finish_reason": "eos" if end is not None else "length",
    }
