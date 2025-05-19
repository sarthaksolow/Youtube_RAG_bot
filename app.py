import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# Your LLM setup here (you must define `model` and `grounded_prompt` before using them)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
grounded_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def answer_youtube_query(video_id: str, query: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return "No captions available for this video."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    new_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    multi_retriever = MultiQueryRetriever.from_llm(retriever=new_retriever, llm=model)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    dynamic_chain = RunnableParallel({
        'context': multi_retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    }) | grounded_prompt | model | StrOutputParser()

    return dynamic_chain.invoke(query)

# Streamlit UI
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎥 YouTube RAG Chatbot")

st.markdown("""
Ask any question based on a YouTube video! Just provide the video ID and a query.
""")

with st.form("query_form"):
    video_id = st.text_input("Enter YouTube Video ID", help="Example: For https://www.youtube.com/watch?v=dQw4w9WgXcQ, enter dQw4w9WgXcQ")
    query = st.text_area("Enter your query", placeholder="Ask something about the video...")
    submit = st.form_submit_button("Get Answer")

if submit:
    if not video_id.strip() or not query.strip():
        st.warning("Please fill in both the video ID and the query.")
    else:
        with st.spinner("Processing... Please wait."):
            try:
                response = answer_youtube_query(video_id.strip(), query.strip())
                st.success("Here's the answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
