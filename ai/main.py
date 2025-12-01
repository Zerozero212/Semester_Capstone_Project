from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import os
import asyncio
import io
import base64

# LangChain 관련
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Gemini 관련
from google import genai
from google.genai import types
from PIL import Image

# 리스트 타입 사용 위해
from typing import List

app = FastAPI(root_path="/ai")

google_api_key = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(
    api_key=google_api_key,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    google_api_key=google_api_key
)

# 스토리 모델(우선 age, topic, words)
class StoryRequest(BaseModel):
    age: int
    topic: str
    words: list[str]

# 동화 기반 문제생성 모델
class ProblemRequest(BaseModel):
    story_text: str # 동화내용 들어가야
    num_questions: int = 5 # 만들 문제의 개수

# 문제 선택지 모델
class ChoiceItem(BaseModel):
    content: str # 선택지 내용
    is_correct: bool # 정답 여부

# 질문 모델
class QuestionItem(BaseModel):
    question: str # 문제 내용
    choices: List[ChoiceItem] # 선택지를 담는 List


# 동화 생성 프롬프트
story_prompt_template = PromptTemplate.from_template(
    """
    You are a professional children's book writer.
    Write a fairy tale based on the inputs.

    [Structure Requirements]
    1. The story MUST be divided into **4 to 6 distinct paragraphs**.
    2. Each paragraph will be one page of the book.
    3. **Output Format:** You MUST return a **JSON list of strings**. Do not include any other text.
       Example: ["Page 1 text...", "Page 2 text...", "Page 3 text..."]

    [Content Instructions]
    - Language: English Only.
    - Translate Korean keywords to English if necessary.
    - Happy ending.
    - Paragraph length: 3~4 sentences per paragraph.

    [Inputs]
    - Target Age: {age} years old
    - Topic: {topic}
    - Required Words: {words}
    """
)

# 문제 생성 프롬프트
problem_prompt_template = PromptTemplate.from_template(
    """
    You are an English education expert for children.
    Based on the provided story, create {num_questions} multiple-choice questions.

    [Story]
    {story_text}

    [Requirements]
    1. Create exactly {num_questions} questions.
    2. Each question must have **5 choices**.
    3. Only **one choice** must be correct (`is_correct`: true).
    4. The questions should test reading comprehension.
    5. Language: English Only.

    [Output Format]
    You MUST return a JSON list of objects matching this exact structure:
    [
      {{
        "question": "Who is the main character?",
        "choices": [
          {{"content": "A Rabbit", "is_correct": true}},
          {{"content": "A Lion", "is_correct": false}},
          {{"content": "A Car", "is_correct": false}},
          {{"content": "A Tree", "is_correct": false}},
          {{"content": "A Bear", "is_correct": false}}
        ]
      }}
    ]
    Do not include any markdown formatting (like ```json). Just return the raw JSON list.
    """
)

# [동기 함수] 실제 SDK를 호출하여 이미지를 만드는 부분
def _generate_image_sync(prompt: str):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=prompt,
        )
        
        # 사용량 로그 출력
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            print(f"이미지 생성 토큰 사용량:")
            print(f"   - 입력 토큰: {usage.prompt_token_count if hasattr(usage, 'prompt_token_count') else 'N/A'}")
            print(f"   - 총 토큰: {usage.total_token_count if hasattr(usage, 'total_token_count') else 'N/A'}")
        
        # 이미지가 inline_data로 반환됨
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # inline_data.data는 bytes 객체이므로 base64 문자열로 변환
                    img_data = part.inline_data.data
                    
                    # bytes인 경우 base64 인코딩
                    if isinstance(img_data, bytes):
                        return base64.b64encode(img_data).decode('utf-8')
                    # 이미 문자열인 경우 그대로 반환
                    return img_data
        
        return None
    except Exception as e:
        print(f"SDK 이미지 생성 중 오류: {e}")
        return None

# [비동기 래퍼] FastAPI가 멈추지 않게 스레드로 실행
async def generate_image_for_page(text: str, index: int, max_retries=2):
    """
    이미지 생성 with 재시도 로직
    """
    # 이미지 프롬프트 (동화 내용을 영어 묘사로 변환)
    image_prompt = f"Create a cute 3D rendered children's book illustration: {text[:300]}"
    
    for attempt in range(max_retries):
        try:
            img_base64 = await asyncio.to_thread(_generate_image_sync, image_prompt)
            
            if img_base64:
                return {
                    "page_no": index + 1,
                    "text": text,
                    "image": img_base64
                }
            
            # 실패 시 재시도 전 대기
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"페이지 {index+1} 시도 {attempt+1} 실패: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    
    return {"page_no": index + 1, "text": text, "image": None}



@app.get("/")
def read_root():
    return {
        "message": "AI Server running with Gemini 2.5 Flash & Free Image Generation",
        "info": {
            "text_model": "gemini-2.5-flash",
            "image_model": "gemini-2.5-flash-image",
            "free_tier_limits": {
                "text": "1500 RPD (Requests Per Day)",
                "image": "500 RPD"
            }
        }
    }

@app.get("/list-models")
def list_available_models():
    try:
        models = client.models.list()
        return {"models": [model.name for model in models]}
    except Exception as e:
        return {"error": str(e)}

# 동화 생성 api 요청 & 함수
@app.post("/generate-story")
async def generate_story(req: StoryRequest):
    text_chain = story_prompt_template | llm | JsonOutputParser()

    try:
        print("동화 텍스트 생성 중...")
        story_pages = await text_chain.ainvoke({
            "age": req.age,
            "topic": req.topic,
            "words": ", ".join(req.words)
        })
        print(f"총 {len(story_pages)}개 페이지 생성 완료")

        final_pages = []
        total_tokens = 0
        
        for i, page_text in enumerate(story_pages):
            print(f"페이지 {i+1}/{len(story_pages)} 이미지 생성 중...")
            page_result = await generate_image_for_page(page_text, i)
            final_pages.append(page_result)
            
            if i < len(story_pages) - 1:
                await asyncio.sleep(2)
        
        print(f"\n전체 동화 생성 완료!")
        print(f"   - 텍스트: {len(story_pages)} 페이지")
        print(f"   - 이미지: {sum(1 for p in final_pages if p['image'])} / {len(story_pages)} 성공")

        result = {
            "title": f"Fairy Tale: {req.topic}",
            "total_pages": len(final_pages),
            "pages": final_pages,
            "preview_url": f"/ai/preview-story?title={req.topic}"  # 미리보기 URL 추가
        }
        
        # 마지막 생성 결과를 메모리에 저장 (미리보기용)
        app.state.last_story = result
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 동화기반 문제 생성 api 요청 & 함수
@app.post("/story-problem", response_model=List[QuestionItem])
async def story_problem(req: ProblemRequest):
    """
    동화 텍스트를 입력받고, 문제를 생성(Question + Choices)
    """
    # 체인 연결
    problem_chain = problem_prompt_template | llm | JsonOutputParser()

    try: 
        print(f"문제 생성 시작 (동화 길이 : {len(req.story_text)}자)")

        # 비동기 호출로 AI에 요청
        result = await problem_chain.ainvoke({
            "story_text" : req.story_text,
            "num_questions" : req.num_questions
        })

        print(f"문제 len{(result)}개 생성 완료!")

        return result
    
    except Exception as e :
        print(f"문제 생성 중 에러 발생 : {e}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/preview-story", response_class=HTMLResponse)
async def preview_story(title: str = "Fairy Tale"):
    """
    마지막 생성된 동화를 HTML로 미리보기
    """
    if not hasattr(app.state, 'last_story') or not app.state.last_story:
        return "<h1>No story generated yet. Please generate a story first.</h1>"
    
    story = app.state.last_story
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{story['title']}</title>
        <style>
            body {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(to bottom, #87CEEB, #98FB98);
            }}
            h1 {{
                text-align: center;
                color: #FF6B6B;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            .page {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .page-number {{
                color: #666;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .text {{
                line-height: 1.8;
                color: #333;
                margin: 15px 0;
            }}
            .image {{
                width: 100%;
                max-width: 512px;
                height: auto;
                border-radius: 10px;
                margin: 15px auto;
                display: block;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .no-image {{
                background: #f0f0f0;
                padding: 40px;
                text-align: center;
                color: #999;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>📚 {story['title']} 📚</h1>
        <p style="text-align: center; color: #666;">Total Pages: {story['total_pages']}</p>
    """
    
    for page in story['pages']:
        html_content += f"""
        <div class="page">
            <div class="page-number">📖 Page {page['page_no']}</div>
            <div class="text">{page['text']}</div>
        """
        
        if page['image']:
            html_content += f"""
            <img class="image" src="data:image/png;base64,{page['image']}" alt="Page {page['page_no']} illustration">
            """
        else:
            html_content += """
            <div class="no-image">🎨 Image generation failed</div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


@app.get("/preview-image/{page_no}", response_class=Response)
async def preview_single_image(page_no: int):
    """
    특정 페이지의 이미지만 PNG로 반환
    """
    if not hasattr(app.state, 'last_story') or not app.state.last_story:
        raise HTTPException(status_code=404, detail="No story found")
    
    story = app.state.last_story
    page = next((p for p in story['pages'] if p['page_no'] == page_no), None)
    
    if not page or not page['image']:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # base64 디코딩하여 이미지 바이너리 반환
    image_data = base64.b64decode(page['image'])
    return Response(content=image_data, media_type="image/png")