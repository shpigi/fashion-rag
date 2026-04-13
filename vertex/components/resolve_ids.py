from kfp import dsl

from fashion_rag.config import COMPONENT_IMAGE


@dsl.component(base_image=COMPONENT_IMAGE)
def resolve_ids(recreate: bool) -> list:
    from fashion_rag.embed import get_ids_to_embed, delete_embeddings_table

    ids = get_ids_to_embed(recreate=recreate)
    if recreate:
        delete_embeddings_table()
    return ids
