from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import create_csv_agent

def main():
    load_dotenv()

    tmp_file_name = "tmp.csv"
    text = ""
    pdf = None
    csv = None
    st.set_page_config(page_title="Ask your file")
    st.header("FileGPT")
    fileType = st.selectbox(
        "Choose the type of file",
        ("", "CSV", "PDF") 
    )
    st.divider()
    
    if fileType:
        icon = ""
        if fileType == "PDF":
            icon = "💬"
        elif fileType == "CSV":
            icon = "📈"
        st.subheader("Ask your {fileType} {icon}".format(fileType=fileType, icon=icon))

        # upload file
        if fileType == "PDF":
            pdf = st.file_uploader("Upload your PDF", type="pdf")
        elif fileType == "CSV":
            csv = st.file_uploader("Upload your CSV", type="csv")

        if csv is not None: 
            text = csv.read().decode("utf-8")
            with open(tmp_file_name, "w") as f:
                f.write(text)

            # create the csv agent
            llm = OpenAI(temperature=0)
            agent = create_csv_agent(llm, tmp_file_name, verbose=True)
            
            # show user input
            st.divider()
            user_question = st.text_input("Ask a question about your CSV.")
            if user_question is not None and user_question != "":
                # send question to llm
                response = agent.run(user_question)
                # Print response to screen
                st.write(response)
            
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split large amount of text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            # create knowledge base
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            st.divider()
            user_question = st.text_input("Ask a question about your PDF.")
            if user_question:
                # docs to pass to llm along with user_question
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(temperature=0)
                chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
                response = chain.run(input_documents=docs, question=user_question)

                # Print response to screen
                st.write(response)

# if application is imported, it will not run
if __name__ == "__main__":
    main()