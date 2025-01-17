from tasks.hf_model import inference, download_model
from flytekit import workflow

# --------------------------------
# Workflow to run inference on multiple queries u
# --------------------------------

@workflow
def wf_text_gen(query: str = "Hello",
                        model_name: str = "microsoft/Phi-3-mini-128k-instruct") -> str:
    model_cache_path = download_model(model_name=model_name)
    result = inference(query=query,
                       model_name=model_name,
                       model_cache_path=model_cache_path)

    return result

#union run --remote workflows/model_infernce.py wf_text_gen --query="what is a fish?"
#union register workflows/model_infernce.py 
