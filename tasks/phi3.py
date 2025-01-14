
from flytekit import workflow, FlyteDirectory, map_task, Artifact
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from functools import partial
import union
from transformers import AutoTokenizer, AutoModel
from typing import Annotated
from containers import actor

from flytekit import ImageSpec
from flytekit import Resources


# --------------------------------
# Load the LLM model as cache for actor
# --------------------------------
@union.actor_cache
def load_model():
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        # cache_dir=cache_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        # cache_dir=cache_path
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# --------------------------------
# Inference task for LLM & Vector DB
# --------------------------------
@actor.task
def inference(query: str ="hello" ) -> str:

  text_gen_pipeline = load_model()

  # Generate a response
  messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": query},
  ]

  generation_args = {
      "max_new_tokens": 500,
      "return_full_text": False,
      "temperature": 0.0,
      "do_sample": False,
  }

  output = text_gen_pipeline(messages, **generation_args)
  llm_response = output[0]['generated_text']

  return llm_response