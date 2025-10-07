from flask import Flask, render_template, jsonify, request, session
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.helper import download_embeddings
from src.prompt import system_prompt
import os
import re
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-this")

# ============================================================================
# CONVERSATION MEMORY STORAGE
# ============================================================================

conversation_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create conversation history for a session"""
    if session_id not in conversation_store:
        conversation_store[session_id] = InMemoryChatMessageHistory()
    return conversation_store[session_id]

# ============================================================================
# EMERGENCY & SAFETY RESPONSES
# ============================================================================

def detect_emergency_query(message: str) -> tuple[bool, str]:
    """Detect emergency/urgent queries and provide appropriate safety responses"""
    message_lower = message.lower().strip()
    
    emergency_patterns = {
        'fever': {
            'keywords': ['fever', 'high temperature', 'hot', 'burning up'],
            'response': """
🌡️ <strong>For Sudden Fever - Immediate Steps:</strong><br><br>
<strong>What to do right now:</strong><br>
• Measure your temperature with a thermometer<br>
• Rest and stay hydrated - drink plenty of water<br>
• Take paracetamol/acetaminophen for fever reduction (follow dosage instructions)<br>
• Wear light, comfortable clothing<br>
• Use a cool compress on your forehead<br><br>
<strong>⚠️ Seek immediate medical attention if:</strong><br>
• Temperature above 103°F (39.4°C)<br>
• Fever lasts more than 3 days<br>
• Accompanied by severe headache, stiff neck, or confusion<br>
• Difficulty breathing or chest pain<br>
• Persistent vomiting or diarrhea<br>
• Rash or unusual symptoms<br><br>
<strong>For children or elderly:</strong> Consult a doctor immediately if fever persists.<br><br>
<em>Note: This is general advice. Always consult healthcare professionals for proper diagnosis and treatment.</em>
            """
        },
        'emergency': {
            'keywords': ['emergency', 'urgent', 'immediately', 'sudden', 'severe pain', 
                        'chest pain', 'difficulty breathing', 'unconscious'],
            'response': """
🚨 <strong>Medical Emergency Detected</strong><br><br>
If this is a life-threatening emergency:<br>
• Call emergency services immediately (ambulance/911)<br>
• Do not wait or try home remedies<br>
• Stay calm and follow emergency dispatcher instructions<br><br>
<strong>Common emergencies requiring immediate attention:</strong><br>
• Chest pain or pressure<br>
• Difficulty breathing<br>
• Severe bleeding<br>
• Loss of consciousness<br>
• Severe allergic reactions<br>
• Signs of stroke (facial drooping, arm weakness, speech difficulty)<br><br>
<em>I'm an AI assistant and cannot provide emergency medical care. Please contact emergency services or visit the nearest emergency room.</em>
            """
        },
        'pain': {
            'keywords': ['severe pain', 'unbearable pain', 'intense pain', 'extreme pain'],
            'response': """
⚠️ <strong>For Severe Pain:</strong><br><br>
<strong>Immediate steps:</strong><br>
• If pain is severe or sudden, seek medical attention immediately<br>
• Do not self-medicate without knowing the cause<br>
• Note the location, intensity, and duration of pain<br>
• Call a doctor or visit urgent care/emergency room<br><br>
<strong>Seek emergency care if pain is:</strong><br>
• In the chest (could be heart-related)<br>
• In the abdomen with vomiting<br>
• Accompanied by fever, confusion, or difficulty breathing<br>
• From an injury or accident<br><br>
<em>Severe pain requires professional medical evaluation. This is not a substitute for emergency care.</em>
            """
        }
    }
    
    for category, data in emergency_patterns.items():
        for keyword in data['keywords']:
            if keyword in message_lower:
                return True, data['response']
    
    return False, ""

# ============================================================================
# GREETING AND CASUAL CONVERSATION DETECTION
# ============================================================================

def detect_greeting_or_casual(message: str) -> tuple[bool, str]:
    """Detect greetings and casual conversation"""
    message_lower = message.lower().strip()
    
    greetings = [
        'hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii', 'helo', 'heyy',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'greetings', 'howdy', 'sup', 'wassup', 'yo'
    ]
    
    casual_patterns = [
        r'^(my name is|i am|im|i\'m)\s+\w+',
        r'^(how are you|how r u|how do you do)',
        r'^(whats up|what\'s up|how\'s it going)',
        r'^(nice to meet you|pleased to meet)',
        r'^(thank you|thanks|thx|ty)',
        r'^(bye|goodbye|see you|see ya|cya)',
        r'^(ok|okay|alright|cool|nice)',
        r'^(yes|no|yeah|yep|nope|sure)',
    ]
    
    if message_lower in greetings or any(message_lower.startswith(g) for g in greetings):
        responses = [
            "Hello! 👋 I'm your Medical AI Assistant. How can I help you today?",
            "Hi there! 😊 I'm here to help with medical information. What would you like to know?",
            "Hey! 👋 I'm a medical chatbot. Feel free to ask me about any health conditions or medical topics!",
        ]
        import random
        return True, random.choice(responses)
    
    if re.match(r'^(my name is|i am|im|i\'m)\s+\w+', message_lower):
        name_match = re.search(r'(my name is|i am|im|i\'m)\s+(\w+)', message_lower)
        if name_match:
            name = name_match.group(2).capitalize()
            return True, f"Nice to meet you, {name}! 😊 I'm your Medical AI Assistant. How can I help you with medical information today?"
    
    for pattern in casual_patterns:
        if re.match(pattern, message_lower):
            return True, "I'm here to help! Do you have any medical questions I can answer? 😊"
    
    return False, ""

def is_medical_question(message: str) -> bool:
    """Check if message contains medical intent"""
    medical_keywords = [
        'disease', 'condition', 'symptom', 'treatment', 'diagnosis', 'doctor',
        'pain', 'fever', 'infection', 'medicine', 'drug', 'surgery', 'therapy',
        'cancer', 'diabetes', 'heart', 'blood', 'pressure', 'virus', 'bacteria',
        'what is', 'how to treat', 'causes of', 'cure', 'remedy', 'health',
        'medical', 'hospital', 'patient', 'illness', 'sick', 'disorder', 
        'how to cure', 'how do i treat', 'treatment for', 'side effects'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in medical_keywords)

# ============================================================================
# INITIALIZE GROQ MODEL
# ============================================================================

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=500,
    timeout=30,
    max_retries=2,
)

# ============================================================================
# INITIALIZE RAG COMPONENTS WITH MEMORY
# ============================================================================

embeddings = download_embeddings()

index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.6}
)

enhanced_system_prompt = """You are a professional medical information assistant. Your role is to provide accurate, evidence-based medical information based on the retrieved context.

**Instructions:**
1. Use ONLY the information provided in the Context section below to answer questions
2. If the context doesn't contain relevant information, clearly state that you don't have that information
3. Provide clear, structured responses with proper medical terminology
4. Always include a disclaimer to consult healthcare professionals
5. For follow-up questions (like "how to cure it?"), use the conversation history to understand what "it" refers to
6. Be empathetic and professional in your responses
7. Format responses with proper structure using bullet points when appropriate

**Important Safety Notes:**
- Never provide emergency medical advice
- Always recommend consulting healthcare professionals for diagnosis and treatment
- Do not replace professional medical consultation

Context: {context}

Answer the user's question based on the above context and conversation history."""

prompt = ChatPromptTemplate.from_messages([
    ("system", enhanced_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(groq_llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"status": "error", "message": "Please provide a message"}), 400
        
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Check for emergency queries
        is_emergency, emergency_response = detect_emergency_query(user_message)
        if is_emergency:
            return jsonify({
                "status": "success",
                "answer": emergency_response,
                "source_count": 0,
                "is_casual": True,
                "is_emergency": True
            })
        
        # Check for greetings
        is_casual, casual_response = detect_greeting_or_casual(user_message)
        if is_casual:
            history = get_session_history(session_id)
            history.add_user_message(user_message)
            history.add_ai_message(casual_response)
            return jsonify({
                "status": "success",
                "answer": casual_response,
                "source_count": 0,
                "is_casual": True
            })
        
        # Check if medical question
        if not is_medical_question(user_message):
            response_msg = ("I'm a medical chatbot designed to provide information about medical conditions, symptoms, and treatments. "
                          "Could you please ask a specific medical question? 😊<br><br>"
                          "<strong>For example:</strong><br>"
                          "• What is diabetes?<br>"
                          "• Symptoms of hypertension<br>"
                          "• How to treat common cold")
            
            history = get_session_history(session_id)
            history.add_user_message(user_message)
            history.add_ai_message(response_msg)
            return jsonify({
                "status": "success",
                "answer": response_msg,
                "source_count": 0,
                "is_casual": True
            })
        
        # Process medical query with RAG + Memory
        response = conversational_rag_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response["answer"]
        context_docs = response.get("context", [])
        formatted_answer = format_response(answer)
        
        if "consult" not in formatted_answer.lower():
            formatted_answer += '<br><br><em>💡 Always consult healthcare professionals for proper diagnosis and treatment.</em>'
        
        return jsonify({
            "status": "success",
            "answer": formatted_answer,
            "source_count": len(context_docs),
            "is_casual": False
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing your request. Please try again."
        }), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history and start fresh"""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            if session_id in conversation_store:
                conversation_store[session_id].clear()
            session['session_id'] = str(uuid.uuid4())
        
        return jsonify({
            "status": "success",
            "message": "Conversation history cleared!"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug/history")
def debug_history():
    """Debug endpoint to view conversation history"""
    if 'session_id' in session:
        history = get_session_history(session['session_id'])
        return jsonify({
            "session_id": session['session_id'],
            "message_count": len(history.messages),
            "messages": [
                {
                    "type": m.__class__.__name__,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content
                }
                for m in history.messages
            ]
        })
    return jsonify({"messages": [], "session_id": None})

def format_response(text: str) -> str:
    """Format response text with proper HTML"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'[•\-]\s+(.+?)(<br>|$|\n)', r'<li>\1</li>', text)
    
    if '<li>' in text:
        text = re.sub(r'(<li>.*?</li>)+', r'<ul style="margin:10px 0;padding-left:20px;">\g<0></ul>', text)
    
    text = text.replace('\n', '<br>')
    return text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
