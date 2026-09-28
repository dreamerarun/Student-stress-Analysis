Student Stress Analyzer v3 🎓🧠

A comprehensive Streamlit-based mental wellness and stress monitoring platform designed for students, doctors, and administrators. The application combines stress assessment, machine learning prediction, AI companionship, consultation scheduling, progress tracking, PDF reporting, email notifications, and Google Calendar integration into a single system.

🌟 Features
👨‍🎓 Student Features
Stress Assessment
20-question academic stress questionnaire
Likert-scale responses
Automatic stress percentage and category detection
Personalized Remedies
Recommendations based on stress level:
Low
Moderate
High
Very High
Progress Tracking
Historical stress records
Interactive graphs
Sleep and health monitoring
PDF Reports
Download stress reports
Email delivery (Google API or SMTP)
AI Companion
Local LLM support via Ollama
Empathetic mental wellness assistant
Conversation history storage
ML Stress Prediction
Predict stress using:
Sleep hours
Study hours
Mood
GPA
Anxiety levels
Support levels
Consultation Requests
Request doctor appointments
Receive meeting links
Google Meet / Jitsi support
👨‍⚕️ Doctor Features
View student consultation requests
Accept / deny appointments
Access student records
View stress history
Join video consultations
Receive email notifications
👨‍💼 Admin Features
Manage users
View system data
Train ML model
Generate datasets
Reset CSV databases
🛠 Technologies Used
Category	Technologies
Frontend	Streamlit
Backend	Python
Data Storage	CSV Files
Machine Learning	Scikit-Learn
Visualization	Plotly
PDF Generation	ReportLab
Sentiment Analysis	TextBlob
Email	Gmail API / SMTP
Video Meetings	Google Meet / Jitsi
AI Companion	Ollama (Llama 3.2)
Authentication	SHA-256 Password Hashing
📂 Project Structure
student_stress_analyzer/
│
├── student_stress_analyzer_v3.py
├── credentials.json               # Google OAuth Credentials
├── data/
│   ├── users.csv
│   ├── progress.csv
│   ├── consultations.csv
│   ├── messages.csv
│   ├── chatbot_responses.csv
│   ├── stress_predictor.pkl
│   ├── academic_stress_dataset.csv
│   ├── google_token.json
│   └── last_user.txt
│
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone Repository
git clone https://github.com/yourusername/student-stress-analyzer.git
cd student-stress-analyzer
2. Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

Or manually:

pip install streamlit pandas numpy scikit-learn plotly textblob reportlab joblib requests google-auth-oauthlib google-api-python-client

Optional:

pip install streamlit-webrtc
📋 Requirements

Create requirements.txt:

streamlit
pandas
numpy
scikit-learn
plotly
reportlab
textblob
joblib
requests
google-auth-oauthlib
google-api-python-client
streamlit-webrtc
🤖 Ollama Setup (AI Companion)

Install Ollama:

https://ollama.com/download

Pull model:

ollama pull llama3.2

Start server:

ollama serve
📧 Email Configuration
Option 1: Gmail API

Place:

credentials.json

in project root.

Google APIs:

Gmail API
Calendar API
Option 2: SMTP

Set environment variables:

export SMTP_SERVER=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your_email@gmail.com
export SMTP_PASS=your_password

Windows:

set SMTP_SERVER=smtp.gmail.com
set SMTP_PORT=587
set SMTP_USER=your_email@gmail.com
set SMTP_PASS=your_password
📅 Google Calendar Integration

Features:

Automatic event creation
Google Meet generation
Student-doctor invitations
Calendar syncing

Fallback:

Jitsi Meet
🧠 Machine Learning Model
Model
RandomForestClassifier
Inputs
Sleep Hours
Study Hours
Blood Pressure
Heart Rate
Mood Score
GPA
Procrastination
Support Level
Anxiety Level
Output
Low Stress
Moderate Stress
High Stress
Very High Stress
📊 Stress Categories
Percentage	Category
0–30%	Low
31–60%	Moderate
61–80%	High
81–100%	Very High
🚀 Run Application
streamlit run student_stress_analyzer_v3.py

Open:

http://localhost:8501
🔒 Security Notes

Current implementation:

✅ Password hashing (SHA-256)

Recommended for production:

PostgreSQL / MySQL
JWT Authentication
HTTPS
Encrypted databases
OAuth Login
Role-based permissions
Docker deployment
📈 Future Improvements
Mobile application
Cloud deployment
Face emotion detection
Wearable integration
Real-time notifications
AI-based personalized therapy plans
Multi-language support
📜 License

This project is intended for educational and research purposes.

👨‍💻 Author

Arun

Developed as an AI-powered academic stress monitoring and student wellness platform integrating:

Machine Learning
Mental Health Analytics
AI Companion Systems
Teleconsultation
Progress Tracking
⭐ Acknowledgements
Streamlit
Scikit-Learn
Plotly
ReportLab
Google APIs
Ollama
TextBlob
Jitsi Meet

Student Stress Analyzer v3 aims to provide early stress detection, personalized support, and accessible consultation services to improve student mental well-being.
