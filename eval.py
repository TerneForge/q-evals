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
from logging import Logger
logger = Logger()

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
        default="kaizen9/phi-1_5_HQ_6000_DONE_20k",
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=["hellaswag, arc_challenge, arc_easy, winogrande, piqa"],
        help="List of evaluation tasks (default: hellaswag).",
    )
    args = parser.parse_args()
    t = get_model()
    r = eval(t, args.tasks, args.num_fewshot)
    logger.log(r)
