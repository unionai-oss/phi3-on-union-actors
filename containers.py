from flytekit import ImageSpec
from union.actor import ActorEnvironment
from flytekit import Resources

image = ImageSpec(
    requirements="requirements.txt",
)

# Create actor environment for Near Real Time serving
# This take a few mins to spin up the inital time
# But then keeps the contain alive unil TTL after last use
actor = ActorEnvironment(
    name="gpu-actor",
    container_image=image,
    replica_count=1,
    ttl_seconds=300,
    requests=Resources(
        cpu="4",
        mem="20000Mi",
        gpu="1", # set different types of GPUs
    ),
)