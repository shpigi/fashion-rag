from kfp import dsl

from fashion_rag.config import COMPONENT_IMAGE


@dsl.component(base_image=COMPONENT_IMAGE)
def evaluate(
    k: int,
    metrics_out: dsl.Output[dsl.Metrics],
    type_confusion_out: dsl.Output[dsl.HTML],
    colour_confusion_out: dsl.Output[dsl.HTML],
    distribution_out: dsl.Output[dsl.HTML],
) -> None:
    import base64
    import io

    from evals.plot_distribution import plot_distribution
    from evals.run_eval_categories import plot_confusion, run_category_eval
    from fashion_rag.search import load_bq_index, load_model

    model, processor = load_model()
    embeddings, metadata = load_bq_index()
    df, summary, type_conf, colour_conf = run_category_eval(
        embeddings, metadata, model, processor, k=k
    )

    metrics_out.log_metric("MRR", summary["mrr"])
    metrics_out.log_metric("MRR_colour", summary["mrr_colour"])
    metrics_out.log_metric("MRR_type", summary["mrr_type"])
    metrics_out.log_metric("Recall", summary["recall"])

    def to_html(title: str, plot_fn, *args) -> str:
        buf = io.BytesIO()
        plot_fn(*args, buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" />'

    with open(type_confusion_out.path, "w") as f:
        f.write(
            to_html(
                "Retrieval confusion by articleType",
                plot_confusion,
                type_conf,
                "articleType (top-5)",
            )
        )

    with open(colour_confusion_out.path, "w") as f:
        f.write(
            to_html(
                "Retrieval confusion by baseColour",
                plot_confusion,
                colour_conf,
                "baseColour (top-5)",
            )
        )

    with open(distribution_out.path, "w") as f:
        f.write(to_html("Dataset distribution", plot_distribution, metadata))
