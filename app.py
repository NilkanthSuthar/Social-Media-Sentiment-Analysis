from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
import streamlit as st
import json

from sentiment import analyze_url, analyze_urls


st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")


def read_uploaded_table(file) -> pd.DataFrame:
    name: str = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    # default to Excel
    return pd.read_excel(file)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return output.getvalue()


def sidebar_controls():
    st.sidebar.title("Settings")
    hf_model_name = st.sidebar.text_input(
        "HuggingFace transformer model",
        value="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="Try 'finiteautomata/bertweet-base-sentiment-analysis' or 'distilbert-base-uncased-finetuned-sst-2-english'",
    )
    neutral_threshold = st.sidebar.slider("Neutral Zone (±)", 0.01, 0.50, 0.05, 0.01)
    auto_model = st.sidebar.checkbox("Auto-pick model based on detected language", value=True)
    st.sidebar.caption("Transformer-based sentiment analysis for social content and web pages.")
    return hf_model_name, float(neutral_threshold), bool(auto_model)


def main():
    st.title("Social Media Sentiment Analyzer")
    st.caption(
        "Enter a social post or article URL. The app extracts the main text and analyzes sentiment using a transformer model."
    )

    hf_model_name, neutral_threshold, auto_model = sidebar_controls()

    url = st.text_input("Enter post URL", placeholder="https://example.com/post-or-article")
    uploaded = st.file_uploader("Or upload a CSV with a 'url' column for batch analysis", type=["csv"], accept_multiple_files=False)

    col1, _ = st.columns([1, 5])
    with col1:
        run = st.button("Analyze URL", type="primary")

    if run:
        if not url and uploaded is None:
            st.warning("Please enter a URL or upload a CSV.")
            return

        if uploaded is not None:
            import pandas as pd
            try:
                df_in = pd.read_csv(uploaded)
                if "url" not in df_in.columns:
                    st.error("CSV must contain a 'url' column")
                    return
                with st.spinner("Batch analyzing URLs..."):
                    df_out = analyze_urls(df_in["url"].astype(str).tolist(), model_name=hf_model_name, neutral_threshold=neutral_threshold, auto_model_by_language=auto_model)
                st.subheader("Batch Results")
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", data=df_out.to_csv(index=False).encode("utf-8"), file_name="url_sentiment.csv", mime="text/csv")
                return
            except Exception as e:
                st.error(f"Batch failed: {e}")
                return

        @st.cache_data(show_spinner=False, ttl=3600)
        def cached_analyze(u: str, model: str, thr: float, auto: bool):
            return analyze_url(
                u,
                model_name=model,
                neutral_threshold=thr,
                auto_model_by_language=auto,
                include_highlights=True,
            )

        with st.spinner("Scraping and analyzing..."):
            try:
                result = cached_analyze(url, hf_model_name, float(neutral_threshold), bool(auto_model))
            except Exception as e:
                st.error(f"Failed: {e}")
                return

        st.subheader("Result")
        st.write(f"URL: {result['url']}")
        if result.get("title"):
            st.write(f"Title: {result['title']}")
        if result.get("language"):
            st.write(f"Language: {result['language']}")
        st.metric("Sentiment", result["Sentiment"], delta=f"Score {result['Score']:.3f}")
        st.write({k: v for k, v in result.items() if k in ["T_POS", "T_NEU", "T_NEG", "MODEL"]})
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "Download JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="url_sentiment.json",
                mime="application/json",
            )
        pos_hl = result.get("highlights_positive", [])
        neg_hl = result.get("highlights_negative", [])
        if pos_hl or neg_hl:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Top positive sentences")
                for s, sc in pos_hl:
                    st.write(f"+ {sc:.3f}: {s}")
            with c2:
                st.caption("Top negative sentences")
                for s, sc in neg_hl:
                    st.write(f"- {sc:.3f}: {s}")

    st.markdown("---")
    with st.expander("About"):
        st.write("Transformer-only sentiment analysis on scraped social content or article text. Features: readability-based extraction, language detection, model selection, sentence highlights, and batch analysis.")


if __name__ == "__main__":
    main()


