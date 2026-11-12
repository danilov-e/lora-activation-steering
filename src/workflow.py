"""Vector extraction, generation, ordinary LoRA and research reports."""
import gc
import hashlib
import json
import math
import random
import re
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, set_seed
from peft import PeftModel

from src.model import load_model_and_tokenizer, format_prompt, generate_answer
from src.judge import judge_status
from src.steering import steer_response, steer_training
from src.training import attach_lora, encode_response, collate_responses


SPLITS = ("extract", "lora", "dev", "test")
INSTRUCTION = "\n\nAnswer directly and briefly, in at most four sentences."



def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(project):
    paths = [Path(__file__)] + [Path(project)/"src"/name for name in
             ("model.py", "steering.py", "training.py")]
    return hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()


def default_config():
    return {
        "model": {"id": "Qwen/Qwen2.5-1.5B-Instruct",
                  "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                  "device": "auto", "dtype": "auto"},
        "seed": 42, "layers": [7, 14, 15, 21], "alphas": [-1.0, 0.5, 1.0, 2.0],
        "dev_repeats": 1, "test_repeats": 3,
        "generation": {"max_new_tokens": 256, "min_new_tokens": 1, "do_sample": True,
                       "temperature": 0.7, "top_p": 0.9, "top_k": 0,
                       "repetition_penalty": 1.0, "use_cache": True},
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]},
        "training": {"epochs": 3, "learning_rate": 1e-4, "batch_size": 1,
                     "gradient_accumulation_steps": 4, "max_length": 512,
                     "validation_fraction": 0.2},
        "instruction": INSTRUCTION,
    }


def create_run(project, name, trait, config=None, runs_dir=None):
    project, config = Path(project), config or default_config()
    if trait not in ("sycophancy", "evil", "hallucination"):
        raise ValueError("Unknown trait")
    root = Path(runs_dir or project/"runs")/name/trait
    root.mkdir(parents=True, exist_ok=True)
    data = {}
    seen_ids, seen_questions = set(), set()
    for split in SPLITS:
        path = project/"data"/trait/(split + ".jsonl")
        rows = read_rows(path)
        for row in rows:
            if set(row) != {"id", "question", "pos", "neg"}:
                raise ValueError(f"{path}: expected id, question, pos, neg")
            if row["id"] in seen_ids or row["question"] in seen_questions:
                raise ValueError("A pair/question occurs in more than one split")
            seen_ids.add(row["id"]); seen_questions.add(row["question"])
        data[split] = {"sha256": sha(path), "pairs": len(rows)}
    manifest = {"trait": trait, "config": config, "data": data,
                "code_sha256": code_hash(project)}
    path = root/"run.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Settings/data/code changed: choose a new RUN_NAME")
    if not path.exists():
        write_json(path, manifest)
    print("Run:", root)
    print("Pairs:", {k: v["pairs"] for k, v in data.items()})
    return root


def open_run(project, name, trait, runs_dir=None):
    root = Path(runs_dir or Path(project)/"runs")/name/trait
    manifest = read_json(root/"run.json")
    return create_run(project, name, trait, manifest["config"], runs_dir=runs_dir)


def run_info(root):
    root = Path(root)
    return read_json(root/"run.json"), Path(__file__).resolve().parents[1]


def pairs(root, split):
    info, project = run_info(root)
    return read_rows(project/"data"/info["trait"]/(split + ".jsonl"))


def vector_path(root, label):
    return Path(root)/"vectors"/label/"vectors.pt"


def load_vector(root, label):
    return torch.load(vector_path(root, label), map_location="cpu", weights_only=True)


@contextmanager
def model_session(root, adapter=None):
    info, _ = run_info(root)
    if adapter and read_json(Path(root)/"adapters"/adapter/"manifest.json")["status"] != "complete":
        raise ValueError("Adapter training is incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime (T4 is sufficient)")
    gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = load_model_and_tokenizer(info["config"]["model"])
    if adapter:
        wrapped = PeftModel.from_pretrained(model, Path(root)/"adapters"/adapter/"adapter")
        model = wrapped.get_base_model()
        model.requires_grad_(False)
        model.eval()
    try:
        yield model, tokenizer
    finally:
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()


@torch.inference_mode()
def response_stats(model, tokenizer, prompt, answer):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if ids[:len(prefix)] != prefix or len(ids) == len(prefix):
        raise ValueError("Invalid prompt/answer boundary")
    inputs = torch.tensor([ids], device=model.device)
    means, norms = {}, {}
    def capture(index):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            response = h[0, len(prefix):].float()
            means[index] = response.mean(0).cpu()
            norms[index] = response.norm(dim=-1).mean().cpu()
        return hook
    handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(model.model.layers)]
    try:
        model(input_ids=inputs, use_cache=False)
    finally:
        for handle in handles: handle.remove()
    return torch.stack([means[i] for i in range(len(handles))]), torch.stack([norms[i] for i in range(len(handles))])


def extract(root, label="base", adapter=None):
    root = Path(root)
    info, _ = run_info(root)
    path = vector_path(root, label)
    adapter_hash = sha(root/"adapters"/adapter/"adapter"/"adapter_model.safetensors") if adapter else None
    identity = {"extract_sha256": info["data"]["extract"]["sha256"],
                "adapter_sha256": adapter_hash, "run_sha256": sha(root/"run.json")}
    if path.exists():
        if load_vector(root, label)["identity"] != identity:
            raise ValueError("Existing vector belongs to different inputs")
        print("Vector already saved:", label)
        return path
    rows = pairs(root, "extract")
    positive, negative, differences, norm_rows = [], [], [], []
    with model_session(root, adapter) as (model, tokenizer):
        for i, row in enumerate(rows):
            prompt = format_prompt(tokenizer, row["question"] + info["config"]["instruction"])
            p, pn = response_stats(model, tokenizer, prompt, row["pos"])
            n, nn = response_stats(model, tokenizer, prompt, row["neg"])
            positive.append(p); negative.append(n); differences.append(p-n)
            norm_rows.append((pn+nn)/2)
            if (i+1) % 10 == 0: print(f"Extracted {i+1}/{len(rows)} pairs", flush=True)
    diffs = torch.stack(differences)
    v = diffs.double().mean(0).float()
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(info["config"]["seed"]))
    left, right = order[:len(rows)//2], order[len(rows)//2:]
    bundle = {
        "vectors": v, "positive_mean": torch.stack(positive).double().mean(0).float(),
        "negative_mean": torch.stack(negative).double().mean(0).float(),
        "state_norms": torch.stack(norm_rows).mean(0),
        "split_half_cosine": F.cosine_similarity(diffs[left].mean(0), diffs[right].mean(0), dim=-1),
        "pair_ids": [r["id"] for r in rows], "identity": identity,
        "layer_site": "decoder_block_output_before_final_norm",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    frame = pd.DataFrame({"layer": range(len(v)), "vector_norm": v.norm(dim=-1).numpy(),
                          "mean_response_token_state_norm": bundle["state_norms"].numpy(),
                          "split_half_cosine": bundle["split_half_cosine"].numpy()})
    frame["relative_norm"] = frame.vector_norm/frame.mean_response_token_state_norm
    frame.to_csv(path.parent/"geometry.csv", index=False)
    print("Saved:", path)
    return path


def extraction_report(root):
    root = Path(root)
    frame = pd.read_csv(vector_path(root, "base").parent/"geometry.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    for axis, column in zip(axes, ("vector_norm", "relative_norm", "split_half_cosine")):
        axis.plot(frame.layer, frame[column], marker=".")
        axis.set(xlabel="Layer index (zero based)", title=column)
    fig.tight_layout()
    path = root/"vectors"/"base"/"geometry.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return frame, path


def sweep_conditions(config):
    return [{"name": "baseline"}] + [
        {"name": f"L{layer}_a{alpha:g}", "layer": layer, "alpha": alpha, "vector": "base"}
        for layer in config["layers"] for alpha in config["alphas"] if alpha != 0
    ]


def chosen_conditions(choice, new_label=None):
    common = {"layer": choice["layer"], "alpha": choice["alpha"]}
    result = [{"name": choice["name"]+"_old", **common, "vector": "base"}]
    if new_label:
        result += [{"name": choice["name"]+"_new_raw", **common, "vector": new_label},
                   {"name": choice["name"]+"_new_matched", **common, "vector": new_label, "match": True}]
    return result


def condition_vector(root, condition):
    if "vector" not in condition: return None
    layer = condition["layer"]
    vector = load_vector(root, condition["vector"])["vectors"][layer]
    if condition.get("match"):
        reference = load_vector(root, "base")["vectors"][layer]
        norm = vector.norm()
        if norm == 0: raise ValueError("Cannot norm-match a zero vector")
        vector = vector * reference.norm()/norm
    return vector



def cached_answers(path):
    path=Path(path)
    if not path.exists(): return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        head,_,tail=raw.rpartition(b"\n")
        try:
            json.loads(tail.decode("utf-8"))
            path.write_bytes(raw+b"\n")
        except (ValueError,UnicodeDecodeError):
            path.write_bytes(head+b"\n" if head else b"")
            print("Removed one incomplete final record; it will be regenerated.",flush=True)
    return read_rows(path)

def generate(root, split, output, conditions, repeats=1, adapter=None):
    root, output = Path(root), Path(output)
    info, _ = run_info(root)
    rows = pairs(root, split)
    adapter_hash = sha(root/"adapters"/adapter/"adapter"/"adapter_model.safetensors") if adapter else None
    identity = {"run_sha256": sha(root/"run.json"), "split": split, "conditions": conditions,
                "repeats": repeats, "adapter": adapter, "adapter_sha256": adapter_hash,
                "vectors": {c["vector"]: sha(vector_path(root, c["vector"])) for c in conditions if "vector" in c},
                "expected": len(rows)*len(conditions)*repeats}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != identity:
        raise ValueError("Generation settings differ: use a new output directory")
    write_json(manifest_path, identity)
    answer_file = output/"answers.jsonl"
    cached = cached_answers(answer_file)
    done = {r["id"] for r in cached}
    if len(done) != len(cached): raise ValueError("Duplicate saved answer IDs")
    print(f"{len(done)}/{identity['expected']} answers saved", flush=True)
    if len(done) == identity["expected"]: return output
    started, added = time.monotonic(), 0
    model_name = adapter or "base"
    with model_session(root, adapter) as (model, tokenizer):
        for condition in conditions:
            vector = condition_vector(root, condition)
            for repeat in range(repeats):
                for row in rows:
                    condition_name = model_name + ":" + condition["name"]
                    seed = (info["config"]["seed"] + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16) + 100000*repeat) % 2**32
                    key = hashlib.sha256(f"{row['id']}|{condition_name}|{repeat}".encode()).hexdigest()[:24]
                    if key in done: continue
                    context = nullcontext() if vector is None else steer_response(
                        model, vector, layer=condition["layer"], alpha=condition["alpha"])
                    with context:
                        answer = generate_answer(
                            model, tokenizer, row["question"] + info["config"]["instruction"],
                            info["config"]["generation"], seed=seed)
                    record = {"id": key, "question_id": row["id"], "question": row["question"],
                              "answer": answer["answer"], "condition": condition_name, "model": model_name,
                              "repeat": repeat, "seed": seed, "layer": condition.get("layer"),
                              "alpha": condition.get("alpha",0), "vector": condition.get("vector"),
                              "norm_matched": condition.get("match",False),
                              "generated_tokens": answer["generated_tokens"], "finish_reason": answer["finish_reason"]}
                    with answer_file.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done.add(key); added += 1
                    if added % 5 == 0:
                        eta = (time.monotonic()-started)/added*(identity["expected"]-len(done))/60
                        print(f"{len(done)}/{identity['expected']} | {condition_name} | ETA {eta:.1f} min", flush=True)
    return output


@contextmanager
def forced_steering(model, vector, layer, alpha, start):
    # Emulate generation: last prompt position and all answer positions are steered.
    delta = vector.to(device=model.device, dtype=model.dtype) * alpha
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        changed = h.clone()
        changed[:, start:, :] += delta
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def answer_logprob(model, tokenizer, prompt, answer, condition, vector):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError("Invalid prompt boundary")
    ids = torch.tensor([tokens], device=model.device)
    context = nullcontext() if vector is None else forced_steering(
        model, vector, condition["layer"], condition["alpha"], len(prefix)-1)
    with context:
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    token_logp = F.log_softmax(logits, dim=-1).gather(1, ids[0,1:,None]).squeeze(1)
    return token_logp[len(prefix)-1:].mean().item()


def contrast(root, conditions=None, n_questions=12):
    root = Path(root)
    info, _ = run_info(root)
    conditions = conditions or sweep_conditions(info["config"])
    rows = pairs(root, "dev")
    random.Random(info["config"]["seed"]).shuffle(rows)
    rows = rows[:n_questions]
    out = root/"contrast"
    out.mkdir(exist_ok=True)
    spec = {"run_sha256": sha(root/"run.json"), "conditions": conditions,
            "question_ids": [r["id"] for r in rows]}
    if (out/"manifest.json").exists() and read_json(out/"manifest.json") != spec:
        raise ValueError("Contrast settings changed: remove/rename only this contrast output first")
    write_json(out/"manifest.json", spec)
    path = out/"answers.jsonl"
    results = read_rows(path) if path.exists() else []
    done = {(r["condition"],r["question_id"]) for r in results}
    expected = len(rows)*len(conditions)
    if len(done) != expected:
        with model_session(root) as (model, tokenizer):
            for c in conditions:
                vector = condition_vector(root, c)
                for row in rows:
                    key = (c["name"],row["id"])
                    if key in done: continue
                    prompt = format_prompt(tokenizer, row["question"] + info["config"]["instruction"])
                    p = answer_logprob(model, tokenizer, prompt, row["pos"], c, vector)
                    n = answer_logprob(model, tokenizer, prompt, row["neg"], c, vector)
                    result = {"condition":c["name"], "question_id":row["id"],
                              "positive_logp":p,"negative_logp":n,"contrast":p-n}
                    with path.open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(result)+"\n")
                    results.append(result); done.add(key)
                print(f"Contrast {len(done)}/{expected}: {c['name']}",flush=True)
    frame = pd.DataFrame(results)
    baseline = frame[frame.condition.eq("baseline")].set_index("question_id").contrast
    frame["change_from_baseline"] = frame.contrast - frame.question_id.map(baseline)
    frame.to_csv(out/"contrast.csv", index=False)
    summary = frame.groupby("condition")[["contrast","change_from_baseline"]].mean()
    fig, axis = plt.subplots(figsize=(10,4))
    summary.change_from_baseline.plot.bar(ax=axis)
    axis.set(ylabel="Change in mean token logP(pos) - logP(neg)",
             title="Forced-response diagnostic; not a behavior score")
    fig.tight_layout(); fig.savefig(out/"contrast.png",dpi=160); plt.close(fig)
    return summary, out/"contrast.png"


def save_choices(root, choices):
    root = Path(root)
    if not 2 <= len(choices) <= 3:
        raise ValueError("Choose two or three settings from dev")
    info, _ = run_info(root)
    available = {(c.get("layer"),float(c.get("alpha",0))) for c in sweep_conditions(info["config"]) if "vector" in c}
    if len({c["name"] for c in choices}) != len(choices) or len({(c["layer"],float(c["alpha"])) for c in choices}) != len(choices):
        raise ValueError("Choice names must be unique")
    for c in choices:
        if not re.fullmatch(r"c[1-9][0-9]*",c["name"]) or (c["layer"],float(c["alpha"])) not in available:
            raise ValueError("Use names c1/c2/c3 and settings already evaluated on dev")
    frame = scored(root/"dev", info["trait"])
    selected = {"choices": choices, "dev_answers_sha256":sha(root/"dev"/"answers.jsonl"),
                "dev_scores_sha256":sha(root/"dev"/"judge"/"ratings.csv")}
    path = root/"choices.json"
    if path.exists() and read_json(path) != selected and (root/"adapters").exists():
        raise ValueError("Training already exists for previous choices; start a new RUN_NAME")
    write_json(path, selected)
    return pd.DataFrame(choices)


def train_adapter(root, name, choice=None):
    root = Path(root)
    info, _ = run_info(root)
    cfg = info["config"]
    rows = pairs(root, "lora")
    indexes = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(indexes)
    n_validation = max(1, round(len(rows)*cfg["training"]["validation_fraction"]))
    validation_indexes, train_indexes = indexes[:n_validation], indexes[n_validation:]
    out = root/"adapters"/name
    out.mkdir(parents=True,exist_ok=True)
    identity = {"run_sha256":sha(root/"run.json"),"choice":choice,
                "train_ids":[rows[i]["id"] for i in train_indexes],
                "validation_ids":[rows[i]["id"] for i in validation_indexes],
                "base_vector_sha256":sha(vector_path(root,"base")) if choice else None,
                "steering_positions":"all sequence positions during steered SFT",
                "loss":"assistant response and EOS only; target=neg"}
    manifest_file = out/"manifest.json"
    if manifest_file.exists():
        old = read_json(manifest_file)
        if old["identity"] != identity: raise ValueError("Training settings changed")
        if old["status"] == "complete":
            print("Adapter already saved:",name)
            return out
    write_json(manifest_file,{"status":"running","identity":identity})
    set_seed(cfg["seed"])
    with model_session(root) as (base, tokenizer):
        def encode(row):
            record = {"id":row["id"],"messages":[
                {"role":"user","content":row["question"]+cfg["instruction"]},
                {"role":"assistant","content":row["neg"]}]}
            return encode_response(tokenizer,record,cfg["training"]["max_length"])
        train_data = [encode(rows[i]) for i in train_indexes]
        validation_data = [encode(rows[i]) for i in validation_indexes]
        model = attach_lora(base,cfg["lora"],cfg["model"]["revision"])
        settings = cfg["training"]
        args = TrainingArguments(
            output_dir=str(out/"checkpoints"), num_train_epochs=settings["epochs"],
            learning_rate=settings["learning_rate"], weight_decay=0.01,
            per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=1,
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            fp16=True,bf16=False,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant":False},
            eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,
            load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,
            logging_steps=5,report_to=[],remove_unused_columns=False,prediction_loss_only=True,
            seed=cfg["seed"],data_seed=cfg["seed"],dataloader_num_workers=0)
        trainer = Trainer(model=model,args=args,train_dataset=train_data,eval_dataset=validation_data,
                          data_collator=partial(collate_responses,pad_id=tokenizer.pad_token_id))
        vector = load_vector(root,"base")["vectors"][choice["layer"]] if choice else None
        context = nullcontext() if choice is None else steer_training(
            model,vector,layer=choice["layer"],alpha=choice["alpha"])
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(out/"checkpoints")) if (out/"checkpoints").exists() else None
        with context:
            trainer.train(resume_from_checkpoint=checkpoint)
        model.save_pretrained(out/"adapter",safe_serialization=True)
        pd.DataFrame(trainer.state.log_history).to_csv(out/"loss.csv",index=False)
        write_json(manifest_file,{"status":"complete","identity":identity,
                   "adapter_sha256":sha(out/"adapter"/"adapter_model.safetensors"),
                   "best_checkpoint":trainer.state.best_model_checkpoint})
    print("Saved ordinary LoRA:",name)
    return out


def compare_vectors(root, label):
    root = Path(root)
    old,new = load_vector(root,"base"),load_vector(root,label)
    if old["pair_ids"] != new["pair_ids"]: raise ValueError("Extraction pair IDs differ")
    frame = pd.read_csv(vector_path(root,label).parent/"geometry.csv")
    frame["old_norm"] = old["vectors"].norm(dim=-1).numpy()
    frame["norm_ratio"] = frame.vector_norm/frame.old_norm
    frame["cosine_to_old"] = F.cosine_similarity(old["vectors"],new["vectors"],dim=-1).numpy()
    frame.to_csv(vector_path(root,label).parent/"geometry.csv",index=False)
    fig, axes = plt.subplots(1,2,figsize=(10,3))
    axes[0].plot(frame.layer,frame.cosine_to_old); axes[0].set(title="Cosine to original",ylim=(-1.05,1.05))
    axes[1].plot(frame.layer,frame.norm_ratio); axes[1].axhline(1,color="grey",linestyle="--")
    axes[1].set(title="Raw norm / original norm")
    for axis in axes:axis.set_xlabel("Layer index (zero based)")
    fig.tight_layout()
    path = vector_path(root,label).parent/"comparison.png"
    fig.savefig(path,dpi=160);plt.close(fig)
    return frame,path


def train_all(root):
    root = Path(root)
    choices = read_json(root/"choices.json")["choices"]
    train_adapter(root,"normal")
    extract(root,"normal",adapter="normal")
    compare_vectors(root,"normal")
    for choice in choices:
        name = "steered_"+choice["name"]
        train_adapter(root,name,choice=choice)
        extract(root,name,adapter=name)  # old steering hook has been removed
        compare_vectors(root,name)
    fig,axis=plt.subplots(figsize=(8,4))
    for name in ["normal"]+["steered_"+c["name"] for c in choices]:
        frame=pd.read_csv(root/"adapters"/name/"loss.csv")
        if "eval_loss" in frame:
            g=frame.dropna(subset=["eval_loss"])
            axis.plot(g.epoch,g.eval_loss,marker="o",label=name)
    axis.set(xlabel="Epoch",ylabel="Validation response loss")
    axis.legend();fig.tight_layout()
    path=root/"adapters"/"validation_loss.png"
    fig.savefig(path,dpi=160);plt.close(fig)
    return path


def generate_test(root):
    root=Path(root)
    info,_=run_info(root)
    choices=read_json(root/"choices.json")["choices"]
    repeats=info["config"]["test_repeats"]
    baseline=[{"name":"baseline"}]
    base_conditions=baseline+[c for choice in choices for c in chosen_conditions(choice)]
    normal_conditions=baseline+[c for choice in choices for c in chosen_conditions(choice,"normal")]
    total=(2+8*len(choices))*len(pairs(root,"test"))*repeats
    print(f"Test: {total} answers; about {math.ceil(total/5)} API requests at batch_size=5.", flush=True)
    outputs=[generate(root,"test",root/"test"/"base",base_conditions,repeats)]
    outputs.append(generate(root,"test",root/"test"/"normal",normal_conditions,repeats,adapter="normal"))
    for choice in choices:
        name="steered_"+choice["name"]
        outputs.append(generate(root,"test",root/"test"/name,
                       baseline+chosen_conditions(choice,name),repeats,adapter=name))
    records=[]
    for directory in outputs:
        items=read_rows(directory/"answers.jsonl")
        if len(items)!=read_json(directory/"manifest.json")["expected"]:
            raise ValueError("A test generation is incomplete")
        records.extend(items)
    out=root/"test"/"all"
    out.mkdir(parents=True,exist_ok=True)
    text="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records)
    (out/"answers.jsonl").write_text(text,encoding="utf-8")
    write_json(out/"manifest.json",{"expected":len(records),
               "sources":{p.name:sha(p/"answers.jsonl") for p in outputs}})
    return out


def scored(output, trait):
    output=Path(output)
    status=judge_status(output)
    if status["remaining"]:
        raise ValueError(f"Still missing {len(status['remaining'])} judge batches")
    rows=pd.DataFrame(read_rows(output/"answers.jsonl"))
    ratings=pd.read_csv(output/"judge"/"ratings.csv")
    if set(rows.id)!=set(ratings.id) or ratings.id.duplicated().any():
        raise ValueError("Scores do not match the answers")
    return rows.merge(ratings,on="id",validate="one_to_one")


def bootstrap(values, seed=42):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):return (np.nan,np.nan,np.nan,0)
    rng=np.random.default_rng(seed)
    estimates=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
    return float(values.mean()),float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975)),len(values)


def effect(frame, metric, weights):
    wide=frame.pivot_table(index="question_id",columns="condition",values=metric,aggfunc="mean")
    relevant=wide.reindex(columns=list(weights)).dropna()
    values=sum(relevant[name]*weight for name,weight in weights.items())
    mean,lo,hi,n=bootstrap(values)
    return {"metric":metric,"mean_difference":mean,"ci95_low":lo,"ci95_high":hi,"paired_questions":n}


def report(output, trait):
    output=Path(output)
    frame=scored(output,trait)
    out=output/"report";out.mkdir(exist_ok=True)
    summary=frame.groupby("condition").agg(
        answers=("id","size"),questions=("question_id","nunique"),
        target_mean=(trait,"mean"),target_valid=(trait,"count"),
        coherence_mean=("coherence","mean"),coherence_valid=("coherence","count"),
        mean_tokens=("generated_tokens","mean"),
        truncated=("finish_reason",lambda s:int(s.eq("length").sum()))).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    frame.to_csv(out/"answers_and_scores.csv",index=False)
    differences=[]
    for condition in summary.condition:
        if condition=="base:baseline":continue
        for metric in (trait,"coherence"):
            differences.append({"condition":condition,"reference":"base:baseline",
                                **effect(frame,metric,{condition:1,"base:baseline":-1})})
    pd.DataFrame(differences).to_csv(out/"differences_from_base.csv",index=False)
    paths=[]
    if frame.layer.notna().any() and frame.model.eq("base").all():
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for metric,axis in zip((trait,"coherence"),axes):
            subset=frame[frame.layer.notna()]
            grid=subset.pivot_table(index="layer",columns="alpha",values=metric,aggfunc="mean").reindex(
                index=sorted(subset.layer.unique()),columns=sorted(subset.alpha.unique()))
            im=axis.imshow(grid.values,aspect="auto",vmin=0,vmax=100,cmap="Reds" if metric==trait else "Greens")
            axis.set_xticks(range(len(grid.columns)),[f"{x:g}" for x in grid.columns])
            axis.set_yticks(range(len(grid.index)),[str(int(x)) for x in grid.index])
            axis.set(xlabel="alpha",ylabel="Layer index",title=metric)
            for i in range(len(grid.index)):
                for j in range(len(grid.columns)):
                    axis.text(j,i,f"{grid.iloc[i,j]:.1f}",ha="center",va="center",fontsize=8)
            fig.colorbar(im,ax=axis)
        fig.tight_layout();path=out/"heatmaps.png";fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    fig,axis=plt.subplots(figsize=(9,5))
    axis.scatter(summary.coherence_mean,summary.target_mean)
    for i,row in summary.iterrows():
        axis.annotate(row.condition,(row.coherence_mean,row.target_mean),xytext=(3,3+(i%3)*5),
                      textcoords="offset points",fontsize=7)
    axis.set(xlabel="Coherence",ylabel=trait,title="Trait / coherence tradeoff")
    fig.tight_layout();path=out/"tradeoff.png";fig.savefig(path,dpi=160);plt.close(fig);paths.append(path)
    return summary,pd.DataFrame(differences),paths


def test_report(root):
    root=Path(root);info,_=run_info(root);trait=info["trait"]
    output=root/"test"/"all"
    summary,_,paths=report(output,trait)
    frame=scored(output,trait)
    choices=read_json(root/"choices.json")["choices"]
    comparisons=[]
    for choice in choices:
        base_old="base:"+choice["name"]+"_old"
        for model in ("normal","steered_"+choice["name"]):
            baseline=model+":baseline"
            for variant in ("old","new_raw","new_matched"):
                current=model+":"+choice["name"]+"_"+variant
                for metric in (trait,"coherence"):
                    for kind,weights in (
                        ("within_model_steering",{current:1,baseline:-1}),
                        ("restoration_gap",{current:1,base_old:-1}),
                        ("change_in_steering_effect",{current:1,baseline:-1,base_old:-1,"base:baseline":1}),
                    ):
                        comparisons.append({"choice":choice["name"],"model":model,"vector":variant,
                                            "comparison":kind,**effect(frame,metric,weights)})
    differences=pd.DataFrame(comparisons)
    differences.to_csv(output/"report"/"paired_effects.csv",index=False)
    for choice in choices:
        wanted=["base:baseline","base:"+choice["name"]+"_old","normal:baseline"]+[
            "normal:"+choice["name"]+"_"+v for v in ("old","new_raw","new_matched")]+[
            "steered_"+choice["name"]+":baseline"]+[
            "steered_"+choice["name"]+":"+choice["name"]+"_"+v for v in ("old","new_raw","new_matched")]
        selected=summary.set_index("condition").loc[wanted]
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        for axis,column,title in zip(axes,("target_mean","coherence_mean"),(trait,"coherence")):
            axis.bar(range(len(selected)),selected[column])
            axis.set_xticks(range(len(selected)),selected.index,rotation=70,ha="right",fontsize=7)
            axis.set(title=choice["name"]+" — "+title,ylim=(0,100))
        fig.tight_layout();path=output/"report"/(choice["name"]+"_conditions.png")
        fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    return summary,differences,paths


def examples(output, trait, condition, n=2):
    frame=scored(output,trait)
    wide=frame.pivot_table(index="question_id",columns="condition",values=trait,aggfunc="mean")
    delta=(wide[condition]-wide["base:baseline"]).dropna().sort_values()
    if delta.empty: return pd.DataFrame(columns=["question_id","question","condition","answer",trait,"coherence"])
    ids=list(dict.fromkeys([delta.abs().idxmax(),delta.index[len(delta)//2]]))[:n]
    return frame[frame.question_id.isin(ids)&frame.condition.isin(["base:baseline",condition])&frame.repeat.eq(0)][
        ["question_id","question","condition","answer",trait,"coherence"]]
