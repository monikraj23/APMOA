import streamlit as st
import pandas as pd

from pathlib import Path

from src.storage.dataset_manager import DatasetManager
from src.storage.knowledge_manager import KnowledgeManager
from src.rag.knowledge_base import APMOAKnowledgeBase


# ============================================================================
# DATASETS
# ============================================================================

def render_datasets_page():

    st.markdown(
        '<div class="section-title">Datasets Management</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Upload, validate and persist event logs for future APMOA runs.'
        '</div>',
        unsafe_allow_html=True,
    )

    manager = DatasetManager()

    # ------------------------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Event Log",
        type=["csv", "xlsx", "xes", "gz"],
        key="dataset_upload",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:

        if st.button(
            "Upload & Validate Dataset",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Storing and validating dataset..."
                ):

                    metadata = manager.save_uploaded_file(
                        uploaded_file
                    )

                st.success(
                    "Dataset uploaded and registered successfully."
                )

                st.json(metadata)

                st.rerun()

            except Exception as e:

                st.error(
                    f"Dataset upload failed: {e}"
                )

    # ------------------------------------------------------------------------
    # REGISTERED DATASETS
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Registered Datasets</div>',
        unsafe_allow_html=True,
    )

    datasets = manager.list_datasets()

    if not datasets:

        st.info(
            "No datasets have been registered yet."
        )

        return

    rows = []

    for dataset in datasets:

        rows.append(
            {
                "Dataset": dataset.get("name", ""),
                "Format": dataset.get("format", ""),
                "Events": dataset.get("events", ""),
                "Cases": dataset.get("cases", ""),
                "Activities": dataset.get("activities", ""),
                "Resources": dataset.get("resources", ""),
                "Status": dataset.get("status", ""),
                "Validated": dataset.get("validated", False),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------------
    # DATASET DETAILS
    # ------------------------------------------------------------------------

    dataset_names = [
        d.get("name", d.get("id", "Unknown"))
        for d in datasets
    ]

    selected_name = st.selectbox(
        "Select dataset",
        dataset_names,
        key="selected_dataset",
    )

    selected_dataset = next(
        (
            d
            for d in datasets
            if d.get("name", d.get("id", "")) == selected_name
        ),
        None,
    )

    if selected_dataset:

        st.markdown(
            '<div class="section-title">Dataset Details</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Events",
            selected_dataset.get("events", "N/A"),
        )

        c2.metric(
            "Cases",
            selected_dataset.get("cases", "N/A"),
        )

        c3.metric(
            "Activities",
            selected_dataset.get("activities", "N/A"),
        )

        c4.metric(
            "Resources",
            selected_dataset.get("resources", "N/A"),
        )

        st.json(selected_dataset)


# ============================================================================
# RAG KNOWLEDGE
# ============================================================================

def render_rag_knowledge_page():

    st.markdown(
        '<div class="section-title">RAG Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Manage persistent knowledge documents used by the APMOA '
        'semantic retrieval and reasoning layer.'
        '</div>',
        unsafe_allow_html=True,
    )

    manager = KnowledgeManager()

    # ------------------------------------------------------------------------
    # INDEX STATUS
    # ------------------------------------------------------------------------

    documents = manager.list_documents()

    indexed_count = sum(
        1
        for document in documents
        if document.get("indexed", False)
    )

    total_count = len(documents)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Documents",
        total_count,
    )

    c2.metric(
        "Indexed",
        indexed_count,
    )

    c3.metric(
        "Pending",
        total_count - indexed_count,
    )

    # ------------------------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Upload Knowledge</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Knowledge Document",
        type=["txt", "md", "pdf", "docx"],
        key="knowledge_upload",
    )

    if uploaded_file is not None:

        if st.button(
            "Upload & Index Knowledge",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Storing document and rebuilding semantic index..."
                ):

                    manager.save_uploaded_file(
                        uploaded_file
                    )

                    knowledge_base = APMOAKnowledgeBase()

                    knowledge_base.build()

                st.success(
                    "Knowledge document uploaded and indexed successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Knowledge indexing failed: {e}"
                )

    # ------------------------------------------------------------------------
    # DOCUMENT TABLE
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Knowledge Documents</div>',
        unsafe_allow_html=True,
    )

    documents = manager.list_documents()

    if not documents:

        st.info(
            "No knowledge documents are registered."
        )

        return

    rows = []

    for document in documents:

        rows.append(
            {
                "Name": document.get("name", ""),
                "Filename": document.get("filename", ""),
                "Format": document.get("format", "").upper(),
                "Status": document.get("status", ""),
                "Indexed": "YES"
                if document.get("indexed", False)
                else "NO",
                "Chunks": document.get("chunks", 0),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------------
    # INDEX CONTROLS
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Index Controls</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Rebuild Semantic Index",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Rebuilding FAISS semantic index..."
                ):

                    knowledge_base = APMOAKnowledgeBase()
                    knowledge_base.build()

                st.success(
                    "Semantic index rebuilt successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Index rebuild failed: {e}"
                )

    with c2:

        if st.button(
            "Refresh Knowledge Status",
            use_container_width=True,
        ):

            st.rerun()

    # ------------------------------------------------------------------------
    # RETRIEVAL TEST
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Semantic Retrieval Test</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Enter a knowledge query",
        placeholder=(
            "Example: What metrics are used to evaluate a process model?"
        ),
        key="rag_test_query",
    )

    if query:

        if st.button(
            "Search Knowledge Base",
            use_container_width=True,
        ):

            try:

                knowledge_base = APMOAKnowledgeBase()
                knowledge_base.build()

                results = knowledge_base.retrieve(
                    query=query,
                    top_k=3,
                )

                if not results:

                    st.warning(
                        "No relevant knowledge was retrieved."
                    )

                else:

                    st.success(
                        f"Retrieved {len(results)} result(s)."
                    )

                    for index, result in enumerate(results):

                        with st.expander(
                            f"{index + 1}. "
                            f"{result.get('source', 'Unknown')} "
                            f"— score "
                            f"{result.get('score', 0):.4f}"
                        ):

                            st.write(
                                result.get("text", "")
                            )

            except Exception as e:

                st.error(
                    f"Retrieval failed: {e}"
                )


# ============================================================================
# EXPERIMENTS
# ============================================================================

def render_experiments_page():

    st.markdown(
        '<div class="section-title">Experiments</div>',
        unsafe_allow_html=True,
    )

    try:

        from src.data.experiment_manager import ExperimentManager

        exp_manager = ExperimentManager()

        exps = exp_manager.get_experiments()

        if exps:

            st.dataframe(
                pd.DataFrame(exps),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No experiments found."
            )

    except ImportError:

        st.warning(
            "ExperimentManager not available."
        )


# ============================================================================
# BASELINE VS APMOA
# ============================================================================

def render_baseline_vs_apmoa():

    st.markdown(
        '<div class="section-title">Baseline vs APMOA</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Comparison functionality goes here."
    )


# ============================================================================
# MULTI-DATASET COMPARISON
# ============================================================================

def render_multi_dataset_comparison():

    st.markdown(
        '<div class="section-title">Multi-Dataset Comparison</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Multi-dataset metrics goes here."
    )


# ============================================================================
# WHAT-IF
# ============================================================================

def render_what_if():

    st.markdown(
        '<div class="section-title">What-If Analysis</div>',
        unsafe_allow_html=True,
    )

    st.slider(
        "Fitness Threshold",
        0.0,
        1.0,
        0.8,
    )

    st.slider(
        "Precision Threshold",
        0.0,
        1.0,
        0.5,
    )


# ============================================================================
# SETTINGS
# ============================================================================

def render_settings():

    st.markdown(
        '<div class="section-title">Settings</div>',
        unsafe_allow_html=True,
    )

    st.checkbox(
        "Enable RAG",
        value=True,
    )

    st.number_input(
        "RAG Top-K",
        value=3,
        min_value=1,
        max_value=10,
    )