from kfp import dsl

from fashion_rag.config import COMPONENT_IMAGE


@dsl.component(base_image=COMPONENT_IMAGE)
def embed_shard(ids: list, shard_index: int, num_shards: int):
    from fashion_rag.embed import embed_images

    embed_images(ids=ids, shard_index=shard_index, num_shards=num_shards)
