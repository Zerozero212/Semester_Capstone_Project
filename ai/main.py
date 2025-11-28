from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# API 통신용 
import httpx

# GEMINI 사용 위해 import
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# 결과 예쁘게
from langchain_core.output_parsers import StrOutputParser
# 함수 체인으로 만드는 도구 RunnableLambda
from langchain_core.runnables import RunnableLambda

import os


app = FastAPI(root_path="/ai")

api_key=os.environ.get("GMS_API_KEY")

# 요청 받을 데이터구조
class StoryRequest(BaseModel):
    age: int
    topic: str
    words: list[str]


# LangChain 설정, 우선 무료 버전인 gpt-5-mini 사용
# 텍스트 생성용 LLM
llm =  ChatOpenAI(
    model = "gpt-5-mini",
    temperature=0.7,
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1",
    api_key=api_key
)

# 이미지 생성용
# ---------------------------------------------------------
# [핵심] imageMaker 정의 (RunnableLambda용 함수)
# ---------------------------------------------------------
async def generate_image_with_gms(story_text: str):
    if not api_key:
        raise ValueError("GMS_API_KEY가 설정되지 않았습니다.")

    # 1. GMS API 호출 설정
    url = f"https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # 동화 내용을 기반으로 이미지 프롬프트 작성
    image_prompt = f"Create a cute, 3d rendered children's book illustration for this story: {story_text[:500]}..."
    
    payload = {
        "contents": [{"parts": [{"text": image_prompt}]}],
        "generationConfig": {
            # [핵심 수정] 예시대로 Text와 Image를 모두 요청해야 합니다!
            "responseModalities": ["Text", "Image"]
        }
    }

    # 2. API 요청
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=30.0)

    img_base64 = None
    
    # 3. 응답 처리
    if response.status_code == 200:
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            
            # parts 리스트 중에서 이미지가 들어있는 부분(inlineData)을 찾습니다.
            for part in parts:
                if "inlineData" in part:
                    img_base64 = part["inlineData"]["data"]
                    break
                    
            if not img_base64:
                print("응답은 왔는데 이미지 데이터가 없습니다.")
                
        except Exception as e:
            print(f"이미지 파싱 실패: {e}")
            print(f"응답 데이터: {data}") # 디버깅용 로그
    else:
        print(f"이미지 생성 실패: {response.text}")

    # 4. 결과 반환
    return {
        "story": story_text,
        "image": img_base64
    }

# 함수를 체인 부품으로 변환
imageMaker = RunnableLambda(generate_image_with_gms)


# Prompt 설정
prompt_template = PromptTemplate.from_template(
    """
    You are a creative fairy tale writer for children.
    Write a short and interesting fairy tale in English based on the following inputs.
    
    [Instructions]
    1. **Language:** The entire story MUST be written in **English**.
    2. **Keyword Translation:** If the 'Required Words' are provided in Korean, **translate them into English** and use the English words in the story. (e.g., '공주' -> 'Princess')
    3. **Content:** It must be hopeful and have a happy ending.
    4. **Length:** 2 to 3 paragraphs.
    
    [Inputs]
    - Target Age: {age} years old
    - Topic: {topic}
    - Required Words: {words}
    
    Story Content:
    """)

@app.get("/")
def read_root():
    return {"message": "Hello from AI Server!"}

@app.post("/generate-story")
async def generate_story(req: StoryRequest):
    # 체인 연결 - 프롬프트 엔지니어링 - 텍스트 생성 - 양식에 맞게 도출 - 텍스트에 맞는 적절한 image 생성
    chain = prompt_template | llm | StrOutputParser() | imageMaker

    try :
        response = await chain.ainvoke({
            "age" : req.age,
            "topic" : req.topic,
            "words" : ", ".join(req.words)
        })

        return response 

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
