from kfp import dsl

from fashion_rag.config import COMPONENT_IMAGE


@dsl.component(base_image=COMPONENT_IMAGE)
def embed_shard(ids_in: dsl.Input[dsl.Artifact], shard_index: int, num_shards: int):
    import json

    from fashion_rag.embed import embed_images

    with open(ids_in.path) as f:
        ids = json.load(f)

    embed_images(ids=ids, shard_index=shard_index, num_shards=num_shards)
