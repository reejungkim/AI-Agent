import streamlit as st
import PyPDF2
import io
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Streamlit 페이지 설정
st.set_page_config(
    page_title="RAG AI-Agent",
    page_icon="🤖",
    layout="wide"
)

# 제목과 설명
st.title("🤖 RAG AI-Agent")
st.markdown("PDF 파일을 업로드하고 내용에 대해 질문해보세요!")

# 사이드바에서 API 키 입력
with st.sidebar:
    st.header("설정")
    openai_api_key = st.text_input(
        "OpenAI API 키를 입력하세요:",
        type="password",
        help="OpenAI API 키가 필요합니다."
    )
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

# 세션 상태 초기화
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def extract_text_from_pdf(pdf_file) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")
        return ""

def create_vectorstore(text: str, api_key: str):
    """텍스트로부터 벡터스토어를 생성합니다."""
    try:
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Document 객체 생성
        documents = [Document(page_content=text)]
        texts = text_splitter.split_documents(documents)
        
        # 임베딩 생성
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # FAISS 벡터스토어 생성
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"벡터스토어 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def create_qa_chain(vectorstore, api_key: str):
    """QA 체인을 생성합니다."""
    try:
        # LLM 설정
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        # 프롬프트 템플릿 설정
        prompt_template = """다음 문맥을 사용하여 질문에 답하세요. 만약 답을 모르겠다면, 모른다고 말하세요. 답을 지어내지 마세요.

{context}

질문: {question}
답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"QA 체인 생성 중 오류가 발생했습니다: {str(e)}")
        return None

# 메인 레이아웃을 두 개의 컬럼으로 구성
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 PDF 파일 업로드")
    
    # PDF 파일 업로드
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type="pdf",
        help="PDF 파일을 업로드하면 내용을 분석하여 질문에 답할 수 있습니다."
    )
    
    # PDF 처리
    if uploaded_file and openai_api_key:
        with st.spinner("PDF를 처리하고 있습니다..."):
            # 텍스트 추출
            text = extract_text_from_pdf(uploaded_file)
            
            if text.strip():
                # 벡터스토어 생성
                vectorstore = create_vectorstore(text, openai_api_key)
                
                if vectorstore:
                    # QA 체인 생성
                    qa_chain = create_qa_chain(vectorstore, openai_api_key)
                    
                    if qa_chain:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        st.success("✅ PDF 처리가 완료되었습니다!")
                        
                        # 문서 정보 표시
                        st.info(f"📊 문서 정보\n- 파일명: {uploaded_file.name}\n- 텍스트 길이: {len(text):,} 문자")
            else:
                st.error("PDF에서 텍스트를 추출할 수 없습니다.")
    
    elif uploaded_file and not openai_api_key:
        st.warning("⚠️ OpenAI API 키를 먼저 입력해주세요.")
    
    # 현재 상태 표시
    if st.session_state.qa_chain:
        st.success("🤖 AI Agent 준비 완료!")
    else:
        st.info("📋 PDF를 업로드하고 API 키를 입력하세요.")

with col2:
    st.header("💬 질문 & 답변")
    
    # 채팅 기록 표시
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # 질문 입력
    if st.session_state.qa_chain:
        if prompt := st.chat_input("PDF 내용에 대해 질문해보세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하고 있습니다..."):
                    try:
                        result = st.session_state.qa_chain({"query": prompt})
                        answer = result["result"]
                        
                        st.write(answer)
                        
                        # 소스 문서 정보 표시 (선택사항)
                        if "source_documents" in result and result["source_documents"]:
                            with st.expander("📚 참고한 문서 부분"):
                                for i, doc in enumerate(result["source_documents"][:2]):
                                    st.write(f"**참고 {i+1}:**")
                                    st.write(doc.page_content[:300] + "...")
                        
                        # 어시스턴트 메시지 추가
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("먼저 PDF 파일을 업로드하고 처리해주세요.")
        
    # 채팅 기록 초기화 버튼
    if st.session_state.messages:
        if st.button("🗑️ 채팅 기록 지우기"):
            st.session_state.messages = []
            st.rerun()

# 하단 정보
st.markdown("---")
st.markdown("""
### 사용 방법:
1. 왼쪽 사이드바에서 **OpenAI API 키**를 입력하세요
2. **PDF 파일**을 업로드하세요
3. PDF 처리가 완료되면 **질문**을 입력하세요
4. AI가 PDF 내용을 바탕으로 **답변**을 제공합니다

### 필요한 라이브러리:
```bash
pip install streamlit langchain langchain-community langchain-openai faiss-cpu PyPDF2 openai
```
""")

# 실행 명령어 안내
st.code("streamlit run app.py", language="bash")