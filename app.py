import streamlit as st

from search import DATA_DIR, encode_text, load_index, load_model, search

IMAGES_DIR = DATA_DIR / "images"

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

METADATA_FIELDS = ["baseColour", "articleType", "season", "gender", "masterCategory"]


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_index():
    return load_index()


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


def main():
    st.set_page_config(page_title="Fashion RAG", layout="wide")
    st.title("Fashion Image Retrieval")

    model, processor = cached_model()
    embeddings, metadata = cached_index()

    k = st.sidebar.slider("Top-K results", min_value=3, max_value=20, value=10)
    mode = st.sidebar.radio("Query mode", ["Free text", "Eval queries"])

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
    results = search(query_emb, embeddings, metadata, k=k)

    if expected:
        match_all = sum(
            1 for _, row in results.iterrows()
            if all(str(row[f]).lower() == v.lower() for f, v in expected.items())
        )
        precision = match_all / len(results)
        st.metric("Precision@K", f"{precision:.0%}")

    cols = st.columns(min(5, k))
    for i, (_, row) in enumerate(results.iterrows()):
        col = cols[i % len(cols)]
        img_path = IMAGES_DIR / f"{row['id']}.jpg"

        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Missing: {row['id']}")

            st.caption(f"Score: {row['score']:.3f}")

            for field in METADATA_FIELDS:
                value = str(row.get(field, ""))
                if expected and field in expected:
                    if value.lower() == expected[field].lower():
                        st.markdown(f":green[{field}: {value}]")
                    else:
                        st.markdown(f":red[{field}: {value}]")
                else:
                    st.text(f"{field}: {value}")


if __name__ == "__main__":
    main()
