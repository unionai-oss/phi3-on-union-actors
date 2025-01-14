from tasks.phi3 import inference
from flytekit import workflow

# --------------------------------
# Workflow to run inference on multiple queries u
# --------------------------------
@workflow
def phi_text_gen_workflow(query: str = "Hello") -> str:

    result = inference(query=query)

    return result

#union run --remote workflows/phi3_response_wf.py phi_text_gen_workflow --query="what is a fish?"