import streamlit as st
from google.cloud import bigquery

from fashion_rag.config import BQ_METADATA_TABLE, GCP_PROJECT, IMAGES_DIR
from fashion_rag.search import encode_text, load_model, search, search_by_id

EVAL_QUERIES = [
    {"query": "red dresses", "expected": {"baseColour": "Red", "articleType": "Dresses"}},
    {"query": "blue jeans", "expected": {"baseColour": "Blue", "articleType": "Jeans"}},
    {"query": "black casual shoes", "expected": {"baseColour": "Black", "articleType": "Casual Shoes"}},
    {"query": "white tshirts", "expected": {"baseColour": "White", "articleType": "Tshirts"}},
    {"query": "black handbags", "expected": {"baseColour": "Black", "articleType": "Handbags"}},
    {"query": "pink tops", "expected": {"baseColour": "Pink", "articleType": "Tops"}},
    {"query": "brown heels", "expected": {"baseColour": "Brown", "articleType": "Heels"}},
    {"query": "green sports shoes", "expected": {"baseColour": "Green", "articleType": "Sports Shoes"}},
    {"query": "black formal shirts", "expected": {"baseColour": "Black", "articleType": "Shirts"}},
    {"query": "yellow kurtas", "expected": {"baseColour": "Yellow", "articleType": "Kurtas"}},
]

METADATA_FIELDS = ["baseColour", "articleType", "gender", "masterCategory"]


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_metadata():
    client = bigquery.Client(project=GCP_PROJECT)
    return client.query(f"SELECT * FROM `{BQ_METADATA_TABLE}`").to_dataframe()


def extract_expected(query, metadata):
    tokens = query.lower().split()
    expected = {}
    for field in ["baseColour", "articleType"]:
        values = metadata[field].dropna().unique()
        for val in values:
            if val.lower() in tokens or all(t in tokens for t in val.lower().split()):
                expected[field] = val
                break
    return expected


def display_results(results, key_prefix="", expected=None):
    cols = st.columns(min(5, len(results)))
    for i, (_, row) in enumerate(results.iterrows()):
        col = cols[i % len(cols)]
        img_path = IMAGES_DIR / f"{row['id']}.jpg"

        with col:
            st.image(img_path.read_bytes(), use_container_width=True)
            parts = []
            for field in METADATA_FIELDS:
                value = str(row.get(field, ""))
                if expected and field in expected:
                    if value.lower() == expected[field].lower():
                        parts.append(f":green[{value}]")
                    else:
                        parts.append(f":red[{value}]")
                else:
                    parts.append(value)
            st.markdown(f"**{row['score']:.3f}** {' / '.join(parts)}")

            st.button("Find similar", key=f"{key_prefix}similar_{i}",
                      on_click=lambda id=int(row["id"]): st.session_state.update(similar_to=id))


def main():
    st.set_page_config(page_title="Fashion RAG", layout="wide")
    st.title("Fashion Image Retrieval")

    model, processor = cached_model()
    metadata = cached_metadata()

    k = st.sidebar.slider("Top-K results", min_value=3, max_value=20, value=10)

    # Similar image mode
    if "similar_to" in st.session_state:
        item_id = st.session_state["similar_to"]
        item = metadata[metadata["id"] == item_id].iloc[0]

        st.subheader(f"Similar to: {item['productDisplayName']}")
        st.button("Back to search",
                  on_click=lambda: st.session_state.pop("similar_to", None))

        results = search_by_id(item_id, k=k)
        display_results(results, key_prefix="sim_")
        return

    # Text search mode
    mode = st.sidebar.radio("Query mode", ["Free text", "Eval queries"])

    expected = None
    if mode == "Eval queries":
        options = {q["query"]: q for q in EVAL_QUERIES}
        selected = st.sidebar.selectbox("Query", list(options.keys()))
        query = selected
        expected = options[selected]["expected"]
    else:
        query = st.text_input("Search for fashion items", placeholder="e.g. red summer dress")
        expected = extract_expected(query, metadata) if query else None

    if not query:
        st.info("Enter a query to search.")
        return

    query_emb = encode_text(query, model, processor)
    results = search(query_emb, k=k)

    if expected:
        match_all = sum(
            1 for _, row in results.iterrows()
            if all(str(row[f]).lower() == v.lower() for f, v in expected.items())
        )
        precision = match_all / len(results)
        st.metric("Precision@K", f"{precision:.0%}")

    display_results(results, key_prefix="txt_", expected=expected)


if __name__ == "__main__":
    main()
