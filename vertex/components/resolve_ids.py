from kfp import dsl

from fashion_rag.config import COMPONENT_IMAGE


@dsl.component(base_image=COMPONENT_IMAGE)
def resolve_ids(recreate: bool, ids_out: dsl.Output[dsl.Artifact]) -> None:
    import json

    from fashion_rag.embed import delete_embeddings_table, get_ids_to_embed

    ids = get_ids_to_embed(recreate=recreate)
    if recreate:
        delete_embeddings_table()

    with open(ids_out.path, "w") as f:
        json.dump(ids, f)
