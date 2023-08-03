import os
import json
import openai
import whisper
import tempfile
import streamlit as st

from tempfile import NamedTemporaryFile
from audiorecorder import audiorecorder

st.set_page_config(
    page_title="Inclusive AGI",
    page_icon="📈",
    layout="wide"
)
# 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
# openai.api_key = st.secrets['API_KEY_OPENAI']
openai.api_key = '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'

def create_prompt(transcript):
    # content = 'Pretend you are an expert in every subject and you have best skills to find key points of the text and summarize text. '
    prompt = f"""Create a list of five important and short points and brief summary of the given text: "{transcript}". Also add the semantic classification of the test, as positive or negative
        Do not include any explanations, only provide a compliant JSON response following this format without deviation."""
    json_prompt = """{"key_points": ["important key points"], "brief_summary": "brief summary", "class_semantic": "class"}"""

    final = prompt + json_prompt

    return final

def openai_create(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": 'Pretend you are an expert in every subject and you have best skills to find key points of the text and summarize text. '}, 
                {"role": "user", "content": prompt}], 
        temperature=0.4, 
        max_tokens=2048,
        frequency_penalty=3, 
        stop=None
    )

    return response['choices'][0]['message']['content']

@st.cache_resource
def process_audio(filename):
    model = whisper.load_model('base')
    result = model.transcribe(filename)

    return result["text"]

st.markdown("<h1 style='text-align: center;'>🤖 Speech-To-Text Summarization</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ - About App", expanded=True):
    st.write(
            """
    -   **Speech-To-Text Summarization** - is an application that takes human speech as input and then output an abridged version as text or speech. 
    -   This sub-application is built around Deep Learning models and OpenAI's ChatGPT.
    	    """
    )
 
st.info('Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV', icon='✨')

with st.sidebar:
    st.error('Note that uploaded files and recorded audios are not stored in any databases and not saved in ChatGPT\'s chat')
    # st.sidebar.write("""---""") 

upload_type = st.sidebar.radio(
        "Choose the uploading type of audio:",
        ('File', 'Record from Browser')
    )

st.sidebar.write("""---""")        

if upload_type == 'File':
    uploaded_file = st.sidebar.file_uploader("Upload audio file: ", type=["wav", "mp3", "ogg", "flac", "mp4"])           

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.write(uploaded_file.getbuffer())
                
        st.markdown("---")

        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(temp_file.name)

        with st.spinner(f"Generating Transcript... 💫"): 
            result = process_audio(temp_file.name)     

        c1, c2 = st.columns([7, 1])                           

        with c1:
            with st.expander("ℹ️ - See Transcript", expanded=False):
                st.markdown(result)

        with c2:
            summary = st.button('Generate Summary')

        if summary:
            # st.write(result)
            with st.spinner(f"Generating Summary... 💫"): 
                final = create_prompt(result)
                # st.write(final)
                response = openai_create(final)

                jsoned = json.loads(response.replace('”', '"'))

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('### Main Points')

                for p in jsoned['key_points']:
                    st.info(p)
            
            with col2:
                st.markdown('### Brief Summary')
                st.success(jsoned['brief_summary'])

else:
    st.write('')
    with st.sidebar:
        recorded = audiorecorder("Start Recording", "Recording...")

    if len(recorded) > 0:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.write(recorded.tobytes())

        st.markdown("---")

        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(temp_file.name)

        with st.spinner(f"Generating Transcript... 💫"): 
            result = process_audio(temp_file.name)     

        c1, c2 = st.columns([7, 1])                           

        with c1:
            with st.expander("ℹ️ - See Transcript", expanded=False):
                st.markdown(result)

        with c2:
            summary = st.button('Generate Summary')

        if summary:
            # st.write(result)
            with st.spinner(f"Generating Summary... 💫"): 
                final = create_prompt(result)
                # st.write(final)
                response = openai_create(final)

                jsoned = json.loads(response.replace('”', '"'))

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('### Main Points')

                for p in jsoned['key_points']:
                    st.info(p)
            
            with col2:
                st.markdown('### Brief Summary')
                st.success(jsoned['brief_summary'])
        

