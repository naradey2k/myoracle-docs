import sys
sys.path.append(".")

import tempfile
import streamlit as st
import time

from PIL import Image
from load_model import *

st.set_page_config(
    page_title="Inclusive AGI",
    page_icon="📈",
    layout="wide"
)

processor, model = load_model()

st.markdown("<h1 style='text-align: center;'>🤖 PicScript</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ - About App", expanded=True):
    st.write(
            """
    -   **Speech-To-Text Summarization** - is an application that takes human speech as input and then output an abridged version as text or speech. 
    -   This sub-application is built around Deep Learning models and OpenAI's ChatGPT.
    	    """
    )

st.info('Supports all popular image formats - PNG, JPG, JPEG', icon='✨')

with st.sidebar:
    st.error('Note that uploaded files are not stored in any databases and not saved in ChatGPT\'s chat')

st.sidebar.write("""---""")    

uploaded_file = st.sidebar.file_uploader("Upload audio file: ", type=["png", "jpg", "jpeg"])           

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_file.write(uploaded_file.getbuffer())
                
    st.markdown("---")

    c1, c2 = st.columns(2, gap='small')                           

    with c1:
        image = Image.open(temp_file.name)
        st.image(image, channels="RGB", width=450, caption='Here is your uploaded image 🖼️')

    with c2:
        question = st.text_input("Ask any question about image to understand the surrounding:")
        gen = st.button('Generate Description')

        if gen:
            answer = get_answer(processor, model, image, question)

            st.markdown("---")
            st.markdown('Here is your description')
            st.success(answer)
