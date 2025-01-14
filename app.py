import gradio as gr
from union.remote import UnionRemote

# Create a remote connection
remote = UnionRemote()

def predict_with_actor(query):

    inputs = {"query": query,}
    
    workflow = remote.fetch_workflow(name="workflows.actor_wf.phi_text_gen_workflow")
    execution = remote.execute(workflow, inputs=inputs, wait=True)

    print(execution.outputs['o0'])
    return execution.outputs['o0']

predict_with_actor("What is a bee?")