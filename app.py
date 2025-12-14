import streamlit as st
import os
import tempfile
import time
from langchain_core.messages import AIMessage, HumanMessage
from rag_engine import RagEngine

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Studio Brain CA",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DARK UI CSS ---
st.markdown("""
<style>
    /* Global Reset & Dark Theme Base */
    .stApp {
        background-color: #0d1117; /* GitHub Dark Dimmed bg */
        color: #c9d1d9; /* Text main */
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #010409; /* Darker sidebar */
        border-right: 1px solid #30363d;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
    }
    
    /* Buttons: Primary (Blue/Action) */
    .stButton > button {
        background-color: #238636; /* Green action */
        color: white;
        border: 1px solid rgba(240,246,252,0.1);
        border-radius: 6px;
        font-weight: 600;
        transition: 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    
    /* Secondary/Ghost Buttons */
    div[data-testid="stHorizontalBlock"] button {
        background-color: #21262d; 
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }

    /* CHAT BUBBLES - CLEAN & DISTINCT */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-msg {
        background: #1f6feb; /* Blue accent */
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        margin-bottom: 12px;
        width: fit-content;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .ai-msg {
        background: #161b22; /* Card bg */
        border: 1px solid #30363d;
        color: #c9d1d9;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 12px;
        width: fit-content;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Library Files */
    .file-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RagEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_view" not in st.session_state:
    st.session_state.current_view = "Chat"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🎹 Studio Brain")
    st.caption("v2.1 | Local Secure Core")
    
    st.markdown("---")
    
    # Custom Nav Menu
    menu = st.radio(
        "Navegación", 
        ["💬 Chat", "📂 Biblioteca"],
        label_visibility="collapsed"
    )
    st.session_state.current_view = menu
    
    st.markdown("---")
    
    # GLOBAL UPLOAD (Available in both views)
    with st.expander("⬆️ Subir Documentos", expanded=False):
        uploaded_files = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            if st.button("Procesar Archivos", type="primary", use_container_width=True):
                progress = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    st.session_state.rag_engine.ingest_pdf(tmp_path, original_filename=file.name)
                    progress.progress((i+1)/len(uploaded_files))
                    os.remove(tmp_path)
                st.success("¡Ingestión completada!")
                time.sleep(1)
                st.rerun()

# --- MAIN AUTO-ROUTING ---
chat_view, lib_view = st.tabs(["Chat Link", "Lib Link"]) 
# NOTE: We use tabs just to keep logic separate or simpler if/else blocks:

if st.session_state.current_view == "💬 Chat":
    
    st.header("💬 Asistente de Producción")
    
    # 1. Chat Area
    chat_box = st.container()
    
    with chat_box:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer

    # 2. Input Area
    if prompt := st.chat_input("Pregunta sobre tus manuales o teoría musical..."):
        # Display User Msg
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Force re-render just to show usage immediately? No, standard flow is fine.
        st.rerun()

    # Handling Response generation after rerun to keep UI snappy
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
         with st.spinner("Analizando manuales..."):
            qa_chain = st.session_state.rag_engine.get_chain()
            if qa_chain:
                # History Bridge
                history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
                
                try:
                    response = qa_chain.invoke({"input": st.session_state.messages[-1]["content"], "chat_history": history})
                except Exception as e:
                    response = f"⚠️ Error: {e}"
            else:
                response = "⚠️ La biblioteca está vacía. Sube documentos en la barra lateral."
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


elif st.session_state.current_view == "📂 Biblioteca":
    st.header("📂 Gestión de Conocimiento")
    st.markdown("Aquí puedes ver y gestionar los manuales que conforman el cerebro de la IA.")
    
    files = st.session_state.rag_engine.get_ingested_files()
    
    if not files:
        st.info("ℹ️ No hay documentos indexados. Usa el panel lateral para subir PDFs.")
    else:
        # Display as a clean grid/list
        st.markdown(f"**{len(files)} Documentos en Memoria**")
        
        for f in files:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"""
                <div class="file-card">
                    <span style="font-weight:600">📄 {f}</span>
                    <span style="color:#8b949e; font-size:0.8em">Indexado</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # Vertical alignment hack or just distinct Delete button
                if st.button("🗑️", key=f"del_{f}", help=f"Eliminar {f}"):
                    res = st.session_state.rag_engine.delete_file(f)
                    st.toast(res)
                    time.sleep(0.5)
                    st.rerun()
