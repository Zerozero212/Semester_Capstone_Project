from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# GEMINI 사용 위해 import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# 결과 예쁘게
from langchain_core.output_parsers import StrOutputParser


app = FastAPI(root_path="/ai")

# 요청 받을 데이터구조
class StoryRequest(BaseModel):
    age: int
    topic: str
    words: list[str]

# LangChain 설정, 우선 무료 버전인 gemini-1.5로 테스트
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)

# Prompt 설정
prompt_template = PromptTemplate.from_template(
    """
    당신은 창의적인 유아용 동화 작가입니다. 
    다음 조건에 맞춰 아이를 위한 짧고 재미있는 동화를 한국어로 지어주세요.
    아이를 위한 동화이기 때문에, 희망적인 메시지가 담겨있고 해피엔딩으로 끝나야 합니다.
    분량은 1문단 당 5 ~ 6 문장으로 구성되며, 총 2 ~ 3 문단이 나와야 합니다.
    
    - 대상 연령: {age}세
    - 주제: {topic}
    - 필수 포함 단어: {words}
    
    동화 내용:                                   
    """
                                               )

@app.get("/")
def read_root():
    return {"message": "Hello from AI Server!"}

@app.post("/generate-story")
async def generate_story(req: StoryRequest):
    # 체인 연결
    chain = prompt_template | llm | StrOutputParser()

    try :
        response = chain.invoke({
            "age" : req.age,
            "topic" : req.topic,
            "words" : ", ".join(req.words)
        })

        return {"story" : response}
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))
