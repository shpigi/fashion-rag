import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("FASHION_API_URL", "http://localhost:8080")

EVAL_QUERIES = [
    {"query": "red dresses", "expected": {"baseColour": "Red", "articleType": "Dresses"}},
    {"query": "blue jeans", "expected": {"baseColour": "Blue", "articleType": "Jeans"}},
    {
        "query": "black casual shoes",
        "expected": {"baseColour": "Black", "articleType": "Casual Shoes"},
    },
    {"query": "white tshirts", "expected": {"baseColour": "White", "articleType": "Tshirts"}},
    {"query": "black handbags", "expected": {"baseColour": "Black", "articleType": "Handbags"}},
    {"query": "pink tops", "expected": {"baseColour": "Pink", "articleType": "Tops"}},
    {"query": "brown heels", "expected": {"baseColour": "Brown", "articleType": "Heels"}},
    {
        "query": "green sports shoes",
        "expected": {"baseColour": "Green", "articleType": "Sports Shoes"},
    },
    {"query": "black formal shirts", "expected": {"baseColour": "Black", "articleType": "Shirts"}},
    {"query": "yellow kurtas", "expected": {"baseColour": "Yellow", "articleType": "Kurtas"}},
]

METADATA_FIELDS = ["baseColour", "articleType", "gender", "masterCategory"]


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30)


@st.cache_data
def cached_metadata_values() -> dict[str, list[str]]:
    resp = api_client().get("/metadata/values")
    resp.raise_for_status()
    return resp.json()


def search_text(query: str, k: int) -> pd.DataFrame:
    resp = api_client().get("/search/text", params={"q": query, "k": k})
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def search_similar(item_id: int, k: int) -> pd.DataFrame:
    resp = api_client().get(f"/search/similar/{item_id}", params={"k": k})
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def extract_expected(query: str, metadata_values: dict[str, list[str]]) -> dict[str, str]:
    tokens = query.lower().split()
    expected: dict[str, str] = {}
    for field in ["baseColour", "articleType"]:
        for val in metadata_values.get(field, []):
            if val.lower() in tokens or all(t in tokens for t in val.lower().split()):
                expected[field] = val
                break
    return expected


def display_results(
    results: pd.DataFrame, key_prefix: str = "", expected: dict[str, str] | None = None
) -> None:
    cols = st.columns(min(5, len(results)))
    for i, (_, row) in enumerate(results.iterrows()):
        col = cols[i % len(cols)]

        with col:
            st.image(f"{API_URL}/images/{row['id']}.jpg", use_container_width=True)
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

            st.button(
                "Find similar",
                key=f"{key_prefix}similar_{i}",
                on_click=lambda id=int(row["id"]): st.session_state.update(similar_to=id),
            )


def main() -> None:
    st.set_page_config(page_title="Fashion RAG", layout="wide")
    st.title("Fashion Image Retrieval")

    metadata_values = cached_metadata_values()

    k = st.sidebar.slider("Top-K results", min_value=3, max_value=20, value=10)

    # Similar image mode
    if "similar_to" in st.session_state:
        item_id = st.session_state["similar_to"]
        results = search_similar(item_id, k=k)
        item_name = results.iloc[0]["productDisplayName"] if not results.empty else str(item_id)

        st.subheader(f"Similar to: {item_name}")
        st.button("Back to search", on_click=lambda: st.session_state.pop("similar_to", None))

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
        expected = extract_expected(query, metadata_values) if query else None

    if not query:
        st.info("Enter a query to search.")
        return

    results = search_text(query, k=k)

    if expected:
        match_all = sum(
            1
            for _, row in results.iterrows()
            if all(str(row[f]).lower() == v.lower() for f, v in expected.items())
        )
        precision = match_all / len(results)
        st.metric("Precision@K", f"{precision:.0%}")

    display_results(results, key_prefix="txt_", expected=expected)


if __name__ == "__main__":
    main()
