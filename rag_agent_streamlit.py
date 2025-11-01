import streamlit as st
import os
import tempfile
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pickle
from pathlib import Path

# Streamlit 페이지 설정
st.set_page_config(
    page_title="RAG AI-Agent (Optimized)",
    page_icon="⚡",
    layout="wide"
)

# 제목과 설명
st.title("⚡ RAG AI-Agent (최적화 버전)")
st.markdown("PDF 파일을 업로드하고 내용에 대해 질문해보세요!")

# 캐시 디렉토리 설정
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# 사이드바에서 API 키 및 설정
with st.sidebar:
    st.header("설정")
    
    # Provider 선택
    provider = st.selectbox(
        "AI Provider 선택",
        ["Groq", "Anthropic", "OpenAI"],
        help="사용할 AI 모델 제공업체를 선택하세요."
    )
    
    # Provider별 API 키 입력
    if provider == "Groq":
        api_key = st.text_input(
            "Groq API 키를 입력하세요:",
            type="password",
            help="Groq API 키가 필요합니다."
        )
    elif provider == "Anthropic":
        api_key = st.text_input(
            "Anthropic API 키를 입력하세요:",
            type="password",
            help="Anthropic API 키가 필요합니다."
        )
    elif provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API 키를 입력하세요:",
            type="password",
            help="OpenAI API 키가 필요합니다."
        )
    
    st.subheader("성능 설정")
    chunk_size = st.slider("청크 크기", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("청크 겹침", 50, 300, 200, 50)
    k_retrieval = st.slider("검색할 문서 수", 2, 10, 3)
    
    # 임베딩 모델 선택
    embedding_model = st.selectbox(
        "임베딩 모델 선택",
        [
            "sentence-transformers/all-MiniLM-L6-v2",  # 가장 빠름
            "sentence-transformers/all-mpnet-base-v2",  # 균형
            "BAAI/bge-small-en-v1.5",  # 작고 빠름
        ],
        index=0
    )
    
    # Provider별 LLM 모델 선택
    if provider == "Groq":
        llm_model = st.selectbox(
            "Groq LLM 모델 선택",
            [
                "llama3-8b-8192",     # 가장 빠름
                "llama3-70b-8192",    # 성능 좋음
                "mixtral-8x7b-32768", # 긴 컨텍스트
            ],
            index=0
        )
    elif provider == "Anthropic":
        llm_model = st.selectbox(
            "Anthropic LLM 모델 선택",
            [
                "claude-3-5-sonnet-20241022",  # 최신 Sonnet
                "claude-3-5-haiku-20241022",   # 빠른 Haiku
                "claude-3-opus-20240229",      # 강력한 Opus
            ],
            index=0
        )
    elif provider == "OpenAI":
        llm_model = st.selectbox(
            "OpenAI LLM 모델 선택",
            [
                "gpt-4o",           # 최신 GPT-4o
                "gpt-4o-mini",      # 빠른 GPT-4o mini
                "gpt-3.5-turbo",    # 빠른 GPT-3.5
            ],
            index=0
        )
    
    # 캐시 관리
    if st.button("캐시 초기화"):
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        st.success("캐시가 초기화되었습니다!")
    
    # API 키 환경변수 설정
    if api_key:
        if provider == "Groq":
            os.environ["GROQ_API_KEY"] = api_key
        elif provider == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key

# 세션 상태 초기화
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

@st.cache_resource
def load_embeddings(model_name: str):
    """임베딩 모델을 캐시하여 로드합니다."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}  # 성능 향상
        )
        return embeddings
    except Exception as e:
        st.error(f"임베딩 모델 로딩 오류: {str(e)}")
        return None

@st.cache_data
def load_pdf_cached(pdf_bytes, filename):
    """PDF 로딩을 캐시합니다."""
    cache_file = CACHE_DIR / f"{filename}_{hash(pdf_bytes)}.pkl"
    
    if cache_file.exists():
        st.info("캐시된 PDF 문서를 로드합니다...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 새로운 PDF 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # 캐시 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)
        
        return documents
    finally:
        os.unlink(tmp_file_path)

def create_vectorstore_optimized(documents: List, embeddings, chunk_size: int, chunk_overlap: int):
    """최적화된 벡터스토어 생성"""
    try:
        if not documents:
            return None, 0
        
        # 문서 해시로 캐시 키 생성
        doc_hash = hash(str([doc.page_content for doc in documents]))
        cache_file = CACHE_DIR / f"vectorstore_{doc_hash}_{chunk_size}_{chunk_overlap}.pkl"
        
        if cache_file.exists():
            st.info("캐시된 벡터스토어를 로드합니다...")
            with open(cache_file, 'rb') as f:
                vectorstore = pickle.load(f)
                return vectorstore, len(documents)
        
        # 새로운 벡터스토어 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        
        # 배치 처리로 임베딩 생성 (메모리 효율성)
        batch_size = 32
        vectorstore = None
        
        progress_bar = st.progress(0)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                temp_vs = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(temp_vs)
            
            progress = (i + batch_size) / len(texts)
            progress_bar.progress(min(progress, 1.0))
        
        progress_bar.empty()
        
        # 캐시 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        return vectorstore, len(texts)
        
    except Exception as e:
        st.error(f"벡터스토어 생성 중 오류: {str(e)}")
        return None, 0

def create_qa_chain_optimized(vectorstore, api_key: str, model_name: str, k: int, provider: str):
    """최적화된 QA 체인 생성"""
    try:
        # Provider별 LLM 설정
        if provider == "Groq":
            llm = ChatGroq(
                temperature=0,
                model_name=model_name,
                groq_api_key=api_key,
                max_tokens=1024,  # 응답 길이 제한으로 속도 향상
            )
        elif provider == "Anthropic":
            llm = ChatAnthropic(
                model=model_name,
                temperature=0,
                max_tokens=1024,
                api_key=api_key,
            )
        elif provider == "OpenAI":
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                max_tokens=1024,
                api_key=api_key,
            )
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}")
        
        # 간단한 프롬프트 (토큰 수 최소화)
        prompt_template = """Context: {context}

Question: {question}
Answer:"""

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"), ])
        
        # 검색기 설정 최적화
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "fetch_k": k * 2  # 더 나은 결과를 위한 오버페칭
            }
        )
        
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     chain_type_kwargs={"prompt": PROMPT},
        #     return_source_documents=True
        # )
        # return qa_chain

        # Create the chains 
        question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
    except Exception as e:
        st.error(f"QA 체인 생성 중 오류: {str(e)}")
        return None

# 메인 레이아웃
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type="pdf",
        help="PDF 파일을 업로드하면 내용을 분석하여 질문에 답할 수 있습니다."
    )
    
    if uploaded_file and api_key:
        # 임베딩 모델 로드
        if st.session_state.embeddings is None:
            with st.spinner("임베딩 모델을 로딩하고 있습니다..."):
                st.session_state.embeddings = load_embeddings(embedding_model)
        
        if st.session_state.embeddings:
            with st.spinner("PDF를 처리하고 있습니다..."):
                start_time = time.time()
                
                # PDF 로드 (캐시됨)
                pdf_bytes = uploaded_file.read()
                documents = load_pdf_cached(pdf_bytes, uploaded_file.name)
                
                if documents:
                    # 벡터스토어 생성 (캐시됨)
                    vectorstore, chunk_count = create_vectorstore_optimized(
                        documents, 
                        st.session_state.embeddings, 
                        chunk_size, 
                        chunk_overlap
                    )
                    
                    if vectorstore:
                        # QA 체인 생성
                        qa_chain = create_qa_chain_optimized(
                            vectorstore, 
                            api_key, 
                            llm_model, 
                            k_retrieval,
                            provider
                        )
                        
                        if qa_chain:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = qa_chain
                            
                            processing_time = time.time() - start_time
                            st.success(f"✅ PDF 처리 완료! ({processing_time:.1f}초)")
                            
                            # 최적화된 문서 정보
                            total_text = sum(len(doc.page_content) for doc in documents)
                            st.info(f"""📊 문서 정보
- 파일명: {uploaded_file.name}
- 페이지 수: {len(documents)}
- 청크 수: {chunk_count}
- 텍스트: {total_text:,} 문자
- 처리 시간: {processing_time:.1f}초""")
                else:
                    st.error("PDF 로딩 실패")
    
    elif uploaded_file and not api_key:
        st.warning(f"⚠️ {provider} API 키를 먼저 입력해주세요.")
    
    # 상태 표시
    if st.session_state.qa_chain:
        st.success("🤖 AI Agent 준비 완료!")
    else:
        st.info("📋 PDF를 업로드하고 API 키를 입력하세요.")

with col2:
    st.header("💬 질문 & 답변")
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # 질문 입력 (API 키가 유효할 때만)
    if st.session_state.qa_chain and api_key:
        if prompt := st.chat_input("PDF 내용에 대해 질문해보세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                start_time = time.time()
                
                with st.spinner("답변 생성 중..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"input": prompt})
                        answer = result.get("answer", "")
                        
                        response_time = time.time() - start_time
                        
                        st.write(answer)
                        st.caption(f"응답 시간: {response_time:.1f}초")
                        
                        # 소스 문서 (축약된 정보)
                        context_docs = result.get("context", [])
                        if context_docs:
                            with st.expander("📚 참고 문서"):
                                for i, doc in enumerate(context_docs[:2]):
                                    page_info = ""
                                    if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                                        page_info = f" (p.{doc.metadata['page'] + 1})"
                                    
                                    st.write(f"**참고 {i+1}{page_info}:**")
                                    st.write(doc.page_content[:200] + "...")
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Provider별 API 에러 메시지 개선
                        if "Error code: 401" in error_msg or "authentication_error" in error_msg or "invalid x-api-key" in error_msg:
                            error_display = f"🔑 **API 키 인증 실패**: {provider} API 키가 올바르지 않습니다. 사이드바에서 정확한 API 키를 다시 입력해주세요."
                        elif "Error code: 429" in error_msg:
                            error_display = f"⏰ **사용량 한도 초과**: {provider} API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
                        elif "Error code: 500" in error_msg:
                            error_display = f"🔧 **서버 오류**: {provider} 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요."
                        elif "anthropic" in error_msg.lower() and ("api_key" in error_msg.lower() or "authentication" in error_msg.lower()):
                            error_display = f"🔑 **API 키 인증 실패**: {provider} API 키가 올바르지 않습니다. 사이드바에서 정확한 API 키를 다시 입력해주세요."
                        elif "openai" in error_msg.lower() and ("api_key" in error_msg.lower() or "authentication" in error_msg.lower()):
                            error_display = f"🔑 **API 키 인증 실패**: {provider} API 키가 올바르지 않습니다. 사이드바에서 정확한 API 키를 다시 입력해주세요."
                        elif "groq" in error_msg.lower() and ("api_key" in error_msg.lower() or "authentication" in error_msg.lower()):
                            error_display = f"🔑 **API 키 인증 실패**: {provider} API 키가 올바르지 않습니다. 사이드바에서 정확한 API 키를 다시 입력해주세요."
                        else:
                            error_display = f"❌ **오류 발생**: {error_msg}"
                        
                        st.error(error_display)
                        st.session_state.messages.append({"role": "assistant", "content": error_display})
    elif not api_key:
        st.info(f"먼저 {provider} API 키를 입력해주세요.")
    else:
        st.info("먼저 PDF 파일을 업로드하고 처리해주세요.")
    
    # 채팅 기록 초기화
    if st.session_state.messages:
        if st.button("🗑️ 채팅 기록 지우기"):
            st.session_state.messages = []
            st.rerun()

# 성능 팁
st.markdown("---")
with st.expander("⚡ 성능 최적화 팁"):
    st.markdown("""
    ### 🚀 속도 향상 방법:
    1. **임베딩 모델**: `all-MiniLM-L6-v2` 사용 (가장 빠름)
    2. **LLM 모델**: 
       - **Groq**: `llama3-8b-8192` (가장 빠름)
       - **Anthropic**: `claude-3-5-haiku-20241022` (빠름)
       - **OpenAI**: `gpt-4o-mini` (빠름)
    3. **청크 크기**: 큰 값(1500-2000)으로 설정하여 청크 수 감소
    4. **검색 문서 수**: 2-3개로 제한
    5. **캐시 활용**: 동일한 PDF는 캐시에서 로드
    
    ### 🎯 현재 최적화 적용 사항:
    - ✅ 임베딩 모델 캐싱 (`@st.cache_resource`)
    - ✅ PDF 처리 결과 캐싱 (`@st.cache_data`)
    - ✅ 벡터스토어 디스크 캐싱
    - ✅ 배치 임베딩 처리
    - ✅ 다중 Provider 지원 (Groq, Anthropic, OpenAI)
    - ✅ 응답 토큰 수 제한
    """)

st.code("streamlit run app.py", language="bash")