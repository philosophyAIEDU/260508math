# 마법의 수학 여행: 미적분 탐험대 🚀

수포자도 이해하는 미분과 적분 인터랙티브 학습 플랫폼입니다. Ollama의 Gemma4 모델을 활용한 친절한 AI 튜터가 함께합니다!

## 📦 설치 방법

1. **저장소 준비 및 파이썬 환경 설정**
   Python 3.11 이상 버전이 필요합니다.
   ```bash
   # 가상환경 생성 (선택사항)
   python -m venv venv
   
   # 가상환경 활성화 (Windows)
   venv\Scripts\activate
   
   # 가상환경 활성화 (Mac/Linux)
   source venv/bin/activate
   ```

2. **의존성 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

## 🤖 Ollama 및 Gemma 모델 설정 방법

1. **Ollama 설치**
   - [Ollama 공식 홈페이지](https://ollama.com/)에 접속하여 운영체제에 맞는 버전을 다운로드하고 설치합니다.

2. **Gemma 모델 다운로드**
   - 터미널(또는 명령 프롬프트)을 열고 다음 명령어를 실행합니다:
   ```bash
   ollama run gemma4:9b
   ```
   - (참고: `gemma4:9b` 모델이 없을 경우 `gemma:7b` 등 사용 가능한 모델로 대체한 뒤 `app.py`의 `MODEL_NAME`을 수정해주세요.)
   - 모델 다운로드가 완료되면 대화창이 뜹니다. `/bye`를 입력해 빠져나옵니다.

3. **Ollama 서버 실행 확인**
   - Ollama 애플리케이션이 백그라운드에서 실행 중인지 확인합니다. (기본적으로 `http://localhost:11434` 에서 동작합니다.)

## ▶️ 실행 명령어

모든 설정이 끝났다면, 아래 명령어로 Streamlit 앱을 실행합니다.
```bash
streamlit run app.py
```
실행 후 브라우저가 자동으로 열리며 앱이 시작됩니다.

## 🛠️ 문제 해결 방법

- **"AI 튜터와 연결할 수 없어요!" (연결 에러 발생 시)**
  - Ollama가 백그라운드에서 실행 중인지 확인하세요.
  - 브라우저에서 `http://localhost:11434`에 접속했을 때 "Ollama is running" 메시지가 나오는지 확인하세요.
- **"모델을 찾을 수 없어요" 에러**
  - 터미널에서 `ollama list`를 입력해 설치된 모델을 확인하세요.
  - 앱에서 사용하는 모델 이름과 일치하는지 확인 후 다르면 `app.py`의 `MODEL_NAME` 변수를 수정하세요.
- **그래프가 보이지 않아요**
  - `plotly` 라이브러리가 정상적으로 설치되었는지 확인하세요. (`pip show plotly`)
