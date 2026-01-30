# Requires transformers>=4.51.0
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# 使用本地模型路径
def get_local_reranker_path():
    """获取本地Reranker模型路径"""
    script_dir = Path(__file__).parent
    local_model_path = script_dir / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-Reranker-0.6B" / "snapshots" / "6e9e69830b95c52b5fd889b7690dda3329508de3"

    # 检查本地模型是否存在且完整
    if local_model_path.exists() and (local_model_path / "config.json").exists():
        config_size = (local_model_path / "config.json").stat().st_size
        if config_size > 100:  # 配置文件应该有一定大小
            print(f"🚀 Using local Qwen3-Reranker model: {local_model_path}")
            return str(local_model_path), True

    print("⚠️ Local reranker model not found or incomplete, using online model...")
    return 'Qwen/Qwen3-Reranker-0.6B', False

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

# 获取模型路径
model_path, is_local = get_local_reranker_path()

# 加载模型
try:
    if is_local:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path).eval()
except Exception as e:
    print(f"❌ Failed to load reranker model from {model_path}: {e}")
    print("🔄 Falling back to online model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B', padding_side='left')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Reranker-0.6B').eval()

print(f"✅ Reranker model loaded successfully")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = ["What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
