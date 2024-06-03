
import streamlit as st

from retriever import PaperRetriever
from config import config


st.set_page_config('VectorarXiv', page_icon=':book:')

@st.cache_resource
def load_retriever():
    return PaperRetriever()

retriever = load_retriever()


st.header(config.title)
st.write(config.subtitle)

user_input = st.text_input(
    "Enter your text:",
    placeholder='帮我找几篇使用 LLM agent 解决软件开发相关问题的文献，最好是最近一年以内'
)

if st.button("提交", type='primary', use_container_width=True):
    print('submitted:', user_input)
    st.session_state['user_input'] = user_input
    st.rerun()

if st.session_state.get('user_input'):
    with st.spinner('Processing...'):
        message_placeholder = st.empty()
        messages = []

        def append_progress_message(message):
            messages.append(message)
            message_placeholder.markdown("\n".join([f"- {msg}" for msg in messages]))
        
        append_progress_message('正在分析问题')
        retriever.launch_retrieval(st.session_state.get('user_input'))

        append_progress_message(f'检索中 (`\"{retriever.state.paper_query.query}\", {retriever.state.paper_query.start_date} ~ {retriever.state.paper_query.end_date}`)')
        retriever.retrieve_results(3, found_new_one_cb=append_progress_message)
    
    # message_placeholder.empty()
    st.success('已完成!')

    st.header('结果')
    for doc, score in retriever.state.all_results:
        if score >= 4:
            st.write(f"{doc.metadata['title']}\n\n"
                     f"- arXiv 标识: [{doc.metadata['id']}](https://arxiv.org/abs/{doc.metadata['id']})\n\n"
                     f"- 文章类别: {doc.metadata['categories']}\n\n")
