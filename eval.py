import lm_eval
import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer
import argparse
import transformers
from safetensors.torch import load_file
from transformers.modeling_utils import load_sharded_checkpoint

import math
from modeling_phi import QPhiForCausalLM

def get_model(model1n, model2n):
    model = QPhiForCausalLM.from_pretrained(
            model1n,
            device_map="cuda",
            torch_dtype = torch.bfloat16
        )
    model.type(torch.bfloat16)
    return model

def eval(model, tasks, shots = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    model = lm_eval.models.huggingface.HFLM(pretrained=model.to(device), batch_size = 8, tokenizer = tokenizer)
    results = lm_eval.simple_evaluate(model = model, tasks = tasks, num_fewshot = shots)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using lm-eval harness.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--merge_model_name",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="Model name (default: gpt2).",
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=["hellaswag"],
        help="List of evaluation tasks (default: hellaswag).",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0 for zero-shot).",
    )
    parser.add_argument(
        "--lossless",
        default=False,
        action="store_true",
        help="Use lossless compression.",
    )
    args = parser.parse_args()
    t = get_model(args.checkpoint_path, args.merge_model_name)
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # t = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    r = eval(t, args.tasks, args.num_fewshot)
    # print("hellaswag acc:", r['results']['hellaswag']['acc_norm,none'])
    # print("piqa:", r['results']['piqa'])
    print("results:", r['results'])
