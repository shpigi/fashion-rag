import argparse

from kfp import compiler, dsl

from fashion_rag.config import GCP_PROJECT, GCP_REGION
from vertex.components.embed_shard import embed_shard
from vertex.components.evaluate import evaluate
from vertex.components.resolve_ids import resolve_ids

NUM_SHARDS = 2
EVAL_K = 10


@dsl.pipeline(name="clip-embed-parallel")
def clip_embed_pipeline(recreate: bool = True) -> None:
    resolve_task = resolve_ids(recreate=recreate)
    # resolve_task.set_caching_options(False)

    with dsl.ParallelFor(
        items=list(range(NUM_SHARDS)),
        parallelism=NUM_SHARDS,
    ) as shard_index:
        embed_task = embed_shard(
            ids_in=resolve_task.outputs["ids_out"],
            shard_index=shard_index,
            num_shards=NUM_SHARDS,
        )
        # embed_task.set_caching_options(False)
        embed_task.set_cpu_limit("4")
        embed_task.set_memory_limit("16G")
        # embed_task.set_accelerator_type("NVIDIA_TESLA_T4")
        # embed_task.set_accelerator_limit(1)

    eval_task = evaluate(k=EVAL_K)
    eval_task.after(embed_task)
    eval_task.set_cpu_limit("4")
    eval_task.set_memory_limit("8G")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args()

    yaml_path = "clip_embed_pipeline.yaml"
    compiler.Compiler().compile(clip_embed_pipeline, yaml_path)
    print(f"Compiled pipeline to {yaml_path}")

    if args.submit:
        from google.cloud import aiplatform

        aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)
        job = aiplatform.PipelineJob(
            display_name="clip-embed",
            template_path=yaml_path,
            parameter_values={"recreate": not args.no_recreate},
        )
        job.submit()
        print(f"Submitted: {job.resource_name}")
    else:
        print("Run with --submit to submit to Vertex AI")
