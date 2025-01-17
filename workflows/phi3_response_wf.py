from tasks.phi3 import inference, download_model
from flytekit import workflow

# --------------------------------
# Workflow to run inference on multiple queries u
# --------------------------------

@workflow
def phi_text_gen_workflow(query: str = "Hello",
                        model_name: str = "microsoft/Phi-3-mini-128k-instruct") -> str:
    model_cache_path = download_model(model_name=model_name)
    result = inference(query=query,
                       model_name=model_name,
                       model_cache_path=model_cache_path)

    return result

#union run --remote workflows/phi3_response_wf.py phi_text_gen_workflow --query="what is a fish?"
