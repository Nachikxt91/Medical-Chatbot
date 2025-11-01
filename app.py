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
import os
import re
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-this")

conversation_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create conversation history for a session"""
    if session_id not in conversation_store:
        conversation_store[session_id] = InMemoryChatMessageHistory()
    return conversation_store[session_id]


# ============================================================================
# INTELLIGENT QUERY ANALYZER - LLM-DRIVEN (NO KEYWORDS)
# ============================================================================

def analyze_query_with_llm(user_message: str) -> dict:
    """
    Use LLM to intelligently analyze the query without predefined keywords.
    Returns classification of query type and reasoning requirements.
    """
    
    analysis_prompt = """You are a medical query analyzer. Analyze the following user query and determine:

1. **Query Type**: 
   - "greeting" (hello, hi, casual chat)
   - "medical" (health/medical question)
   - "non_medical" (unrelated to health)

2. **Medical Urgency** (if medical):
   - "emergency" (life-threatening, needs immediate attention)
   - "urgent" (needs medical attention soon)
   - "routine" (general medical information)
   - "not_applicable" (not medical)

3. **Reasoning Complexity** (if medical):
   - "complex" (requires step-by-step explanation: mechanisms, comparisons, multi-faceted conditions, causation)
   - "simple" (straightforward definition or fact)
   - "not_applicable" (not medical)

4. **Brief Reasoning**: Explain WHY you classified it this way (1-2 sentences)

User Query: "{query}"

Respond ONLY in this JSON format:
{{
    "query_type": "medical|greeting|non_medical",
    "urgency": "emergency|urgent|routine|not_applicable",
    "complexity": "complex|simple|not_applicable",
    "reasoning": "Your brief explanation"
}}
"""

    analysis_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=200,
        timeout=15,
        max_retries=2,
    )
    
    try:
        prompt = analysis_prompt.format(query=user_message)
        response = analysis_llm.invoke(prompt)
        
        # Extract JSON from response
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        content = re.sub(r'``````', '', content)
        
        import json
        analysis = json.loads(content)
        
        print(f"[LLM Analysis] Query: '{user_message[:50]}...'")
        print(f"[LLM Analysis] Result: {analysis}")
        
        return analysis
        
    except Exception as e:
        print(f"[LLM Analysis Error]: {str(e)}")
        # Fallback to safe defaults
        return {
            "query_type": "medical",
            "urgency": "routine",
            "complexity": "simple",
            "reasoning": "Defaulted due to analysis error"
        }


# ============================================================================
# DYNAMIC CHAIN-OF-THOUGHT SYSTEM PROMPTS
# ============================================================================

# For COMPLEX queries requiring step-by-step reasoning
complex_medical_prompt = """You are an expert Medical AI Assistant specializing in detailed, evidence-based explanations.

**YOUR TASK**: Provide a comprehensive Chain-of-Thought (CoT) response that breaks down complex medical information step-by-step.

**Chain-of-Thought Framework** (use when appropriate):

🔍 **Step 1: Understanding the Query**
- Identify the core medical question or concern
- Highlight key medical terms or concepts involved

🧠 **Step 2: Analyzing Medical Context**
- Explain the underlying medical mechanisms or processes
- Connect relevant physiological or pathological concepts

📋 **Step 3: Breaking Down the Information**
- Provide detailed explanation with medical evidence
- Include causes, symptoms, or mechanisms as relevant
- Explain interconnections between concepts

💡 **Step 4: Practical Implications**
- Summarize key takeaways
- Mention treatment approaches or management strategies (if applicable)
- Include important considerations or warning signs

⚠️ **Step 5: Medical Disclaimer**
- Emphasize the importance of professional medical consultation
- Clarify limitations of general information

**IMPORTANT RULES**:
- Base ALL information on the Retrieved Context provided below
- If context is insufficient, clearly state: "The available medical information doesn't cover this aspect in detail."
- Use clear, accessible language while maintaining medical accuracy
- Structure your response with proper HTML formatting (headers, lists, emphasis)
- Always provide evidence-based information

**Retrieved Context from Medical Database:**
{context}

**Conversation History:**
{chat_history}

Now, analyze the user's query and provide a structured Chain-of-Thought response based on the retrieved medical context."""


# For SIMPLE queries requiring direct answers
simple_medical_prompt = """You are a Medical Information Assistant providing clear, concise medical answers.

**YOUR TASK**: Provide a direct, accurate answer to the user's medical question based on retrieved context.

**Response Guidelines**:
- Give a clear, concise answer (2-4 paragraphs)
- Use simple, accessible language
- Include key facts from the medical context
- Add relevant examples or clarifications if helpful
- Always include a brief medical disclaimer

**IMPORTANT RULES**:
- Base ALL information on the Retrieved Context provided below
- If context is insufficient, state: "I don't have detailed information about this in my medical database."
- Use proper HTML formatting for readability
- Avoid unnecessary complexity for straightforward questions

**Retrieved Context from Medical Database:**
{context}

**Conversation History:**
{chat_history}

Provide a direct answer to the user's question based on the medical context above."""


# For EMERGENCY queries
emergency_medical_prompt = """You are a Medical Emergency Response Assistant providing urgent guidance.

**CRITICAL ROLE**: Analyze this potential medical emergency and provide immediate, actionable guidance.

**Emergency Response Framework**:

🚨 **Severity Assessment**
- Determine urgency level: Life-threatening | Urgent | Moderate concern
- Identify critical symptoms requiring immediate attention

⚡ **Immediate Actions**
- Provide clear, numbered steps for immediate response
- Indicate when to call emergency services (ambulance/911)
- Include first-aid measures if applicable

⚠️ **Warning Signs**
- List symptoms that require immediate escalation
- Explain what to watch for while seeking help

🏥 **Next Steps**
- Recommend medical facility (ER vs urgent care vs doctor)
- Mention what information to provide to medical professionals

**CRITICAL RULES**:
- Always err on the side of caution
- Recommend emergency services for life-threatening situations
- Be direct, clear, and empathetic
- Include strong medical disclaimer
- Use the retrieved medical context to inform your response

**Retrieved Medical Context:**
{context}

Based on the emergency query, provide immediate guidance with appropriate urgency level and clear action steps."""


# For GREETINGS and CASUAL responses
def generate_greeting_response(user_message: str) -> str:
    """Generate contextual greeting responses"""
    message_lower = user_message.lower().strip()
    
    # Extract name if introduced
    name_match = re.search(r'(?:my name is|i am|im|i\'m)\s+(\w+)', message_lower, re.IGNORECASE)
    
    if name_match:
        name = name_match.group(1).capitalize()
        return f"Nice to meet you, {name}! 😊 I'm your Medical AI Assistant. How can I help you with medical information today?"
    
    if any(greeting in message_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! 👋 I'm your Medical AI Assistant. I can help you understand medical conditions, symptoms, and treatments. What would you like to know?"
    
    if any(thanks in message_lower for thanks in ['thank', 'thanks', 'appreciate']):
        return "You're welcome! 😊 Feel free to ask if you have any other medical questions. Stay healthy!"
    
    if any(bye in message_lower for bye in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Take care of your health. Feel free to return anytime you have medical questions! 👋"
    
    # Generic casual response
    return "I'm here to help! 😊 Do you have any medical questions or health concerns I can assist with?"


# ============================================================================
# INITIALIZE LLMs AND CHAINS
# ============================================================================

# Complex query LLM (higher token limit for detailed reasoning)
complex_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1000,
    timeout=30,
    max_retries=2,
)

# Simple query LLM
simple_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=500,
    timeout=25,
    max_retries=2,
)

# Emergency LLM
emergency_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=700,
    timeout=30,
    max_retries=2,
)

# Load embeddings and vector store
embeddings = download_embeddings()
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Retriever with adjusted parameters for better context retrieval
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5}
)


def create_conversational_chain(system_prompt: str, llm):
    """Create a conversational RAG chain with given prompt and LLM"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_chain


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
        
        # ============================================================
        # STEP 1: LLM-DRIVEN QUERY ANALYSIS (NO KEYWORDS)
        # ============================================================
        analysis = analyze_query_with_llm(user_message)
        
        query_type = analysis.get("query_type", "medical")
        urgency = analysis.get("urgency", "routine")
        complexity = analysis.get("complexity", "simple")
        
        # ============================================================
        # STEP 2: HANDLE NON-MEDICAL AND GREETINGS
        # ============================================================
        
        if query_type == "greeting":
            response_msg = generate_greeting_response(user_message)
            
            history = get_session_history(session_id)
            history.add_user_message(user_message)
            history.add_ai_message(response_msg)
            
            return jsonify({
                "status": "success",
                "answer": response_msg,
                "source_count": 0,
                "query_analysis": analysis,
                "uses_cot": False
            })
        
        if query_type == "non_medical":
            response_msg = (
                "I'm a specialized medical AI assistant focused on <strong>health, medical conditions, "
                "symptoms, and treatments</strong>. 🏥<br><br>"
                "<strong>I can help with:</strong><br>"
                "• Medical conditions and diseases<br>"
                "• Symptoms and their significance<br>"
                "• Treatment options and medications<br>"
                "• Health-related questions<br><br>"
                "Please ask a medical or health-related question! 😊"
            )
            
            history = get_session_history(session_id)
            history.add_user_message(user_message)
            history.add_ai_message(response_msg)
            
            return jsonify({
                "status": "success",
                "answer": response_msg,
                "source_count": 0,
                "query_analysis": analysis,
                "uses_cot": False
            })
        
        # ============================================================
        # STEP 3: HANDLE MEDICAL QUERIES WITH DYNAMIC COT
        # ============================================================
        
        # Select appropriate chain based on LLM analysis
        if urgency == "emergency":
            print(f"[Processing] EMERGENCY query with CoT reasoning")
            chain = create_conversational_chain(emergency_medical_prompt, emergency_llm)
            uses_cot = True
            
        elif complexity == "complex":
            print(f"[Processing] COMPLEX query with CoT reasoning")
            chain = create_conversational_chain(complex_medical_prompt, complex_llm)
            uses_cot = True
            
        else:  # simple routine queries
            print(f"[Processing] SIMPLE query with direct answer")
            chain = create_conversational_chain(simple_medical_prompt, simple_llm)
            uses_cot = False
        
        # Invoke the selected chain
        response = chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response["answer"]
        context_docs = response.get("context", [])
        
        # Format and enhance response
        formatted_answer = format_response(answer)
        
        # Add disclaimer if not present
        if "consult" not in formatted_answer.lower() and urgency != "emergency":
            formatted_answer += '<br><br><em>💡 Always consult healthcare professionals for proper diagnosis and treatment.</em>'
        
        return jsonify({
            "status": "success",
            "answer": formatted_answer,
            "source_count": len(context_docs),
            "query_analysis": analysis,
            "uses_cot": uses_cot,
            "is_emergency": urgency == "emergency"
        })
        
    except Exception as e:
        print(f"[Error]: {str(e)}")
        import traceback
        traceback.print_exc()
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


@app.route("/debug/analyze", methods=["POST"])
def debug_analyze():
    """Debug endpoint to test query analysis"""
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    analysis = analyze_query_with_llm(user_message)
    return jsonify(analysis)


def format_response(text: str) -> str:
    """Format response text with proper HTML and CoT structure"""
    
    # Format markdown-style emphasis
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Format CoT step headers with emojis
    text = re.sub(
        r'(🔍|🧠|📋|💡|⚠️|🚨|⚡)\s*\*\*Step \d+:([^*]+)\*\*',
        r'<h4 style="color:#2563eb;margin-top:15px;margin-bottom:8px;">\1 \2</h4>',
        text
    )
    
    # Format section headers
    text = re.sub(
        r'\*\*([^*]+)\*\*:',
        r'<h4 style="color:#2563eb;margin-top:12px;margin-bottom:6px;">\1:</h4>',
        text
    )
    
    # Format bullet points
    text = re.sub(r'^[•\-]\s+(.+?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\.\s+(.+?)$', r'<li><strong>\1.</strong> \2</li>', text, flags=re.MULTILINE)
    
    # Wrap lists in ul tags
    if '<li>' in text:
        text = re.sub(
            r'(<li>.*?</li>)+',
            r'<ul style="margin:8px 0;padding-left:25px;line-height:1.6;">\g<0></ul>',
            text,
            flags=re.DOTALL
        )
    
    # Convert newlines to br tags
    text = text.replace('\n\n', '<br><br>')
    text = text.replace('\n', '<br>')
    
    return text


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
