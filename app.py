from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
from src.helper import download_embeddings
from src.prompt import system_prompt
from typing import Any, List, Optional, Dict
import os
import re

load_dotenv()

app = Flask(__name__)

# ============================================================================
# EMERGENCY & SAFETY RESPONSES
# ============================================================================

def detect_emergency_query(message: str) -> tuple[bool, str]:
    """
    Detect emergency/urgent queries and provide appropriate safety responses
    """
    message_lower = message.lower().strip()
    
    # Emergency keywords
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
            'keywords': ['emergency', 'urgent', 'immediately', 'sudden', 'severe pain', 'chest pain', 'difficulty breathing', 'unconscious'],
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
    
    # Check for emergency patterns
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
        'medical', 'hospital', 'patient', 'illness', 'sick', 'disorder'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in medical_keywords)

# ============================================================================
# ENHANCED MEDICAL LLM WITH STRICT CONTEXT CHECKING
# ============================================================================

class CleanMedicalLLM(LLM):
    """Medical LLM with strict context relevance checking"""
    
    @property
    def _llm_type(self) -> str:
        return "clean_medical_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            context = ""
            question = ""
            
            if "Context:" in prompt:
                context_start = prompt.find("Context:") + len("Context:")
                if "Human:" in prompt:
                    context_end = prompt.find("Human:")
                    context = prompt[context_start:context_end].strip()
                    human_start = prompt.find("Human:") + len("Human:")
                    question = prompt[human_start:].strip()
                else:
                    context = prompt[context_start:].strip()
                    question = context
            else:
                context = prompt
                question = prompt
            
            # Check if context is meaningful and sufficient
            if not context or len(context.strip()) < 50:
                return self._format_no_context_response()
            
            context_clean = context.replace('\n', ' ').replace('  ', ' ').strip()
            
            # Extract key terms from question
            question_lower = question.lower()
            question_keywords = set(re.findall(r'\b\w+\b', question_lower))
            
            # Check if context is relevant to the question
            context_lower = context_clean.lower()
            
            # Remove common stop words for better matching
            stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'do', 'does', 'should', 'can', 'the', 'a', 'an', 'for', 'to', 'of', 'in', 'on'}
            relevant_keywords = question_keywords - stop_words
            
            # Check keyword overlap
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in context_lower)
            relevance_score = keyword_matches / len(relevant_keywords) if relevant_keywords else 0
            
            # If relevance is too low, return no context response
            if relevance_score < 0.3:  # At least 30% keyword match required
                return self._format_no_context_response()
            
            # Check for medical content
            medical_keywords = [
                'disease', 'condition', 'disorder', 'characterized', 'caused',
                'results', 'symptoms', 'treatment', 'occurs', 'involves',
                'affects', 'leads to', 'develops', 'manifests', 'presents',
                'diagnosed', 'associated', 'commonly', 'typically', 'syndrome',
                'medical', 'patient', 'clinical', 'therapy', 'diagnosis'
            ]
            
            has_medical_content = any(keyword in context_lower for keyword in medical_keywords)
            
            if not has_medical_content:
                return self._format_no_context_response()
            
            # Extract meaningful sentences
            sentences = context_clean.split('.')
            medical_sentences = []
            
            skip_patterns = [
                'System', 'You are', 'Context:', 'Human:', 'GEM -', 'Page ',
                'GALE ENCYCLOPEDIA', 'Harrison', 'ed.', 'Reproduced by permission',
                'Copyright', 'All rights reserved', 'ISBN'
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                should_skip = any(pattern in sentence for pattern in skip_patterns)
                
                if (len(sentence) > 30 and not should_skip and
                    not sentence.endswith(':') and not sentence.startswith('"')):
                    
                    # Prioritize sentences matching question keywords
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in relevant_keywords):
                        medical_sentences.insert(0, sentence)  # Add to front
                    elif any(keyword in sentence_lower for keyword in medical_keywords):
                        medical_sentences.append(sentence)
                
                if len(medical_sentences) >= 5:
                    break
            
            if medical_sentences:
                response = self._format_medical_response(medical_sentences[:4])
                return response
            
            return self._format_no_context_response()
            
        except Exception as e:
            return self._format_error_response()
    
    def _format_medical_response(self, sentences: List[str]) -> str:
        """Format medical sentences with proper structure"""
        formatted_text = '. '.join(sentences) + '.'
        formatted_text = formatted_text.replace('  ', ' ').strip()
        
        # Add line breaks between sentences for readability
        formatted_text = re.sub(r'(\. )([A-Z])', r'.<br><br>\2', formatted_text)
        
        # Add disclaimer
        formatted_text += '<br><br><em>💡 Always consult healthcare professionals for proper diagnosis and treatment.</em>'
        
        return formatted_text
    
    def _format_no_context_response(self) -> str:
        """Response when information is not in context"""
        return ("⚠️ <strong>Information Not Available</strong><br><br>"
                "I apologize, but I don't have specific information about this topic in my current medical knowledge base.<br><br>"
                "<strong>Recommendations:</strong><br>"
                "• Consult with a healthcare professional for accurate information<br>"
                "• Visit a doctor or urgent care if you have health concerns<br>"
                "• Try asking about common medical conditions<br><br>"
                "<em>For urgent medical issues, please seek immediate medical attention.</em>")
    
    def _format_error_response(self) -> str:
        """Response when an error occurs"""
        return ("❌ <strong>Processing Error</strong><br><br>"
                "I encountered an error while processing your request. "
                "Please try asking your question in a different way.")

# ============================================================================
# INITIALIZE RAG COMPONENTS WITH SIMILARITY THRESHOLD
# ============================================================================

embeddings = download_embeddings()

index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever with similarity score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 6,  # Retrieve more documents
        "score_threshold": 0.5  # Only return docs with >50% similarity
    }
)

chatModel = CleanMedicalLLM()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({
                "status": "error",
                "message": "Please provide a message"
            }), 400
        
        # 1. Check for emergency/urgent queries FIRST
        is_emergency, emergency_response = detect_emergency_query(user_message)
        if is_emergency:
            return jsonify({
                "status": "success",
                "answer": emergency_response,
                "source_count": 0,
                "is_casual": True,
                "is_emergency": True
            })
        
        # 2. Check for greetings or casual conversation
        is_casual, casual_response = detect_greeting_or_casual(user_message)
        if is_casual:
            return jsonify({
                "status": "success",
                "answer": casual_response,
                "source_count": 0,
                "is_casual": True
            })
        
        # 3. Check if it's a medical question
        if not is_medical_question(user_message):
            return jsonify({
                "status": "success",
                "answer": ("I'm a medical chatbot designed to provide information about medical conditions, symptoms, and treatments. "
                          "Could you please ask a specific medical question? 😊<br><br>"
                          "<strong>For example:</strong><br>"
                          "• What is diabetes?<br>"
                          "• Symptoms of hypertension<br>"
                          "• How to treat common cold"),
                "source_count": 0,
                "is_casual": True
            })
        
        # 4. Process medical query with RAG
        response = rag_chain.invoke({"input": user_message})
        answer = response["answer"]
        context_docs = response.get("context", [])
        
        # Format answer for better display
        formatted_answer = format_response(answer)
        
        return jsonify({
            "status": "success",
            "answer": formatted_answer,
            "source_count": len(context_docs),
            "is_casual": False
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing your request. Please try again."
        }), 500

def format_response(text: str) -> str:
    """Format response text with proper HTML"""
    # Convert markdown-style formatting to HTML
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert bullet points
    text = re.sub(r'•\s+(.+?)(<br>|$)', r'<li>\1</li>', text)
    if '<li>' in text:
        text = '<ul style="margin:10px 0;padding-left:20px;">' + text + '</ul>'
    
    return text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
