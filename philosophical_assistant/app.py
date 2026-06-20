import streamlit as st
from src.rag.llm import ask

# Общие настройки
st.set_page_config( page_title='Philosophy Assistant',
                    page_icon=':open_book:',layout='wide'
                    )

# Заголовок и описание
st.title('Philosophy Assistant')
st.markdown(""" Ask questions about philosophy, philosophers 
                and philosophical concepts.
    
                You can ask a question in your preferred language 
                and receive an answer in the same language.
            """)

# Поле для ввода вопроса
question = st.text_input('Question')

if st.button('Ask'):

    # Вывод при отсутствии вопроса
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner('Searching corpus...'):

        # Получение ответа от модели
        try:
            answer, sources = ask(question)

        # Вывод сообщения об ошибке при отсутствии связи с моделью
        except Exception as e:
            st.error(
                f"Error while generating answer:\n{e}"
            )

    # Вывод ответа модели
    st.subheader('Answer')
    st.write(answer)

    # Вывод источников
    st.subheader('Sources')
    st.dataframe(sources,use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []