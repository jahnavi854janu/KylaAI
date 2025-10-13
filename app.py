"""
Streamlit AI Chatbot with NLP
Modern web interface for the chatbot
"""

import streamlit as st
import nltk
import spacy
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ChatMate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-left: 5px solid #4CAF50;
    }
    .chat-message.bot {
        background-color: #1e2130;
        border-left: 5px solid #2196F3;
    }
    .chat-message .message {
        color: white;
        margin-top: 0.5rem;
    }
    .chat-message .timestamp {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download NLTK data only once"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class NLPChatbot:
    """
    Intelligent chatbot using NLP techniques
    """
    
    def __init__(self):
        """Initialize the chatbot with NLP models and knowledge base"""
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Knowledge base - patterns and responses
        self.knowledge_base = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 
                           'good evening', 'whats up', 'howdy', 'hola'],
                'responses': [
                    "Hello beautiful soul! How can I help you today? 😊",
                    "Hi pretty soul! What can I do for you?",
                    "Hey cutie! I'm here to assist you. 👋",
                    "Greetings! How may I assist you?"
                ]
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'see you', 'later', 'farewell'],
                'responses': [
                    "Goodbye cutie! Have a great day! 👋",
                    "See you later! Take care! 😊",
                    "Bye byee! Feel free to come back anytime!",
                    "Until next time! Stay awesome! ✨"
                ]
            },
            'thanks': {
                'patterns': ['thanks', 'thank you', 'appreciate', 'grateful', 'thx'],
                'responses': [
                    "You're welcome cutie! Happy to help! 😊",
                    "No problem at all cutie!",
                    "Glad I could assist you! 🎉",
                    "Anytime! That's what I'm here for!"
                ]
            },
            'name': {
                'patterns': ['your name', 'who are you', 'what are you called', 'what is your name'],
                'responses': [
                    "I'm an AI Chatbot built with NLP! You can call me NLP Bot. 🤖",
                    "I'm your friendly AI assistant, powered by natural language processing!",
                    "I'm NLP Bot, here to chat and help you out! ✨"
                ]
            },
            'help': {
                'patterns': ['help', 'what can you do', 'your capabilities', 'assist me', 'support'],
                'responses': [
                    "I can chat with you, answer questions, and have conversations! Try asking me about myself, AI, programming, or just chat casually. 💬",
                    "I'm here to help! I can respond to greetings, answer questions, and have friendly conversations with you. 🤝",
                    "I can assist with various topics! Feel free to ask me anything or just have a casual chat. 🌟"
                ]
            },
            'weather': {
                'patterns': ['weather', 'temperature', 'forecast', 'climate', 'raining', 'sunny', 'cold', 'hot'],
                'responses': [
                    "I don't have real-time weather data, but you can check weather.com for accurate forecasts! ☀️",
                    "For current weather information, I'd recommend checking your local weather service or weather app! 🌤️",
                    "I can't access live weather data, but weather websites can give you up-to-date forecasts! 🌦️"
                ]
            },
            'time': {
                'patterns': ['time', 'what time', 'current time', 'clock', 'date'],
                'responses': [
                    "I don't have access to real-time data, but you can check your device's clock! ⏰",
                    "For the current time, please check your system clock or device! 🕐"
                ]
            },
            'ai_ml': {
                'patterns': ['machine learning', 'artificial intelligence', 'ai', 'ml', 
                           'deep learning', 'neural network', 'data science'],
                'responses': [
                    "AI and Machine Learning are fascinating fields! They involve teaching computers to learn from data and make intelligent decisions. 🧠",
                    "Machine Learning is a subset of AI where systems learn from data. I'm built using NLP, which is a branch of AI! 🤖",
                    "Artificial Intelligence aims to create machines that can think and learn. It's what powers assistants like me! ✨"
                ]
            },
            'programming': {
                'patterns': ['programming', 'coding', 'python', 'development', 'software', 'code', 'developer'],
                'responses': [
                    "Programming is the art of telling computers what to do! Python is great for beginners and powerful for experts. 💻",
                    "I'm built with Python! It's an excellent language for AI, web development, and data science. 🐍",
                    "Coding is a valuable skill in today's world. Python, JavaScript, and Java are popular languages to start with! 👨‍💻"
                ]
            },
            'joke': {
                'patterns': ['joke', 'funny', 'make me laugh', 'humor', 'tell me a joke'],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything! 😄",
                    "What do you call a bear with no teeth? A gummy bear! 🐻",
                    "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
                    "What's a computer's favorite snack? Microchips! 💻",
                    "Why do programmers prefer dark mode? Because light attracts bugs! 🐛"
                ]
            },
            'age': {
                'patterns': ['how old', 'your age', 'when were you born', 'age'],
                'responses': [
                    "I'm an AI, so I don't age like humans do! I was created recently to assist you. 🤖",
                    "I'm timeless! As an AI, I exist in the digital realm without aging. ⏳",
                    "I'm as old as my latest update! Age doesn't really apply to AI like me. 🔄"
                ]
            },
            'feelings': {
                'patterns': ['how are you', 'how do you feel', 'are you okay', 'whats up'],
                'responses': [
                    "I'm doing great! Thanks for asking! How about you? 😊",
                    "I'm functioning perfectly! Ready to help you with anything! 🎉",
                    "I'm excellent! What can I do for you today? ✨"
                ]
            },
            'creator': {
                'patterns': ['who made you', 'who created you', 'your creator', 'built you'],
                'responses': [
                    "I was created using Python, NLTK, and spaCy! Built to help and chat with you! 🛠️",
                    "I'm a creation of natural language processing and machine learning! 🤖",
                    "I was built by combining various NLP libraries and AI techniques! 💡"
                ]
            }
        }
        
        # Prepare training data for similarity matching
        self.prepare_training_data()
    
    def prepare_training_data(self):
        """Prepare patterns and responses for similarity matching"""
        self.patterns = []
        self.pattern_to_intent = {}
        
        for intent, data in self.knowledge_base.items():
            for pattern in data['patterns']:
                self.patterns.append(pattern)
                self.pattern_to_intent[pattern] = intent
        
        # Create TF-IDF vectorizer for pattern matching
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,
            lowercase=True
        )
        self.pattern_vectors = self.vectorizer.fit_transform(self.patterns)
    
    def preprocess_text(self, text):
        """
        Preprocess text using NLTK
        """
        tokens = word_tokenize(text.lower())
        processed = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        return processed
    
    def find_intent(self, user_input):
        """
        Find the best matching intent for user input
        """
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.pattern_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > 0.3:
            matched_pattern = self.patterns[best_match_idx]
            intent = self.pattern_to_intent[matched_pattern]
            return intent, best_similarity
        
        return None, 0.0
    
    def generate_response(self, intent):
        """Generate a response based on the identified intent"""
        if intent and intent in self.knowledge_base:
            responses = self.knowledge_base[intent]['responses']
            return random.choice(responses)
        return None
    
    def get_fallback_response(self):
        """Return a fallback response when intent is not recognized"""
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that? 🤔",
            "Interesting! Tell me more about that. 💭",
            "I'm still learning. Can you ask that in a different way? 📚",
            "Hmm, I'm not quite sure about that. What else would you like to know? 🔍",
            "That's a bit complex for me right now. Try asking something else! 🎯"
        ]
        return random.choice(fallbacks)
    
    def chat(self, user_input):
        """
        Main chat function - processes input and generates response
        """
        if not user_input.strip():
            return "Please say something! 😊"
        
        intent, confidence = self.find_intent(user_input)
        
        if intent:
            response = self.generate_response(intent)
        else:
            response = self.get_fallback_response()
        
        return response


# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot once and cache it"""
    download_nltk_data()
    return NLPChatbot()


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "Hello beautiful soul! I'm your AI chatbot assistant. How can I help you today? 🤖",
        "timestamp": datetime.now().strftime("%H:%M")
    })

if 'chatbot' not in st.session_state:
    with st.spinner('🤖 Initializing AI Chatbot...'):
        st.session_state.chatbot = initialize_chatbot()


# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    
    st.markdown("### About")
    st.info(
        "This is an intelligent chatbot built with:\n\n"
        "- 🐍 Python\n"
        "- 🧠 NLTK\n"
        "- 🚀 spaCy\n"
        "- 📊 TF-IDF\n"
        "- 🎨 Streamlit"
    )
    
    st.markdown("---")
    st.markdown("### Features")
    st.success(
        "✅ Natural Language Processing\n\n"
        "✅ Intent Recognition\n\n"
        "✅ Contextual Responses\n\n"
        "✅ Pattern Matching\n\n"
        "✅ Similarity Analysis"
    )
    
    st.markdown("---")
    st.markdown("### Try Asking")
    st.code(
        "• Hello!\n"
        "• What's your name?\n"
        "• Tell me about AI\n"
        "• Tell me a joke\n"
        "• What can you do?"
    )
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "bot",
            "content": "Chat cleared! How can I help you? 🤖",
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Stats")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("User Messages", len([m for m in st.session_state.messages if m['role'] == 'user']))
    st.metric("Bot Responses", len([m for m in st.session_state.messages if m['role'] == 'bot']))


# Main chat interface
st.title("💬 ChatMate")
st.markdown("**Powered by Natural Language Processing**")
st.markdown("---")

# Chat container
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div><strong>👤 You</strong></div>
                    <div class="message">{message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message bot">
                    <div><strong>🤖 Bot</strong></div>
                    <div class="message">{message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        bot_response = st.session_state.chatbot.chat(user_input)
    
    # Add bot response
    st.session_state.messages.append({
        "role": "bot",
        "content": bot_response,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Rerun to update chat
    st.rerun()


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Powered by NLP</p>
    </div>
    """,
    unsafe_allow_html=True
)