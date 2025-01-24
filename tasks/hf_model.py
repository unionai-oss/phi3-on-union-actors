
from flytekit import workflow, FlyteDirectory, map_task, Artifact
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from functools import partial
import union
from transformers import AutoTokenizer, AutoModel
from typing import Annotated
from containers import actor, image

from flytekit import task , current_context  
from flytekit import Resources
from pathlib import Path

# --------------------------------
# Download LLM model from Hugging Face
# --------------------------------
@task(
    container_image=image,
    cache=True,
    cache_version="0.007",
    enable_deck=True,
    requests=Resources(cpu="2", mem="8Gi"),
)
def download_model(model_name: str) -> FlyteDirectory:
    
    working_dir = Path(current_context().working_directory)
    saved_model_dir = working_dir / "saved_model"
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

    return FlyteDirectory(saved_model_dir)

# --------------------------------
# Load the LLM model as cache for actor
# --------------------------------
@union.actor_cache
def load_model(model_name:str, model_dir: FlyteDirectory) -> pipeline:
    # Load the model and tokenizer
    model_dir = model_dir.download()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# --------------------------------
# Inference task for LLM & Vector DB
# --------------------------------
@actor.task(enable_deck=True,)
def inference(query: str, model_name: str, model_dir: FlyteDirectory ) -> str:

  text_gen_pipeline = load_model(model_name=model_name, model_dir=model_dir)

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
