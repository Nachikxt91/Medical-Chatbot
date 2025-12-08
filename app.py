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
import json

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
# IMPROVED QUERY ANALYZER WITH BETTER CLASSIFICATION
# ============================================================================

def analyze_query_with_llm(user_message: str) -> dict:
    """
    Use LLM to intelligently analyze the query with improved classification.
    Returns classification of query type and reasoning requirements.
    """
    
    analysis_prompt = """
You are a **medical query classifier** for a Medical AI Assistant.

Analyze the user's message and return a JSON object with:

1. "query_type":
   - "greeting"  → greetings, thanks, goodbyes, casual small talk
   - "medical"   → health, symptoms, diseases, treatments, medications, lifestyle and prevention
   - "non_medical" → anything not related to health or medicine

2. "urgency" (only if query_type = "medical"):
   - "emergency" → possible life‑threatening issues (for example: chest pain, stroke signs, suicidal thoughts, severe breathing difficulty, heavy bleeding, major trauma)
   - "urgent" → needs attention soon but not obviously life‑threatening (for example: very high fever, severe pain, new serious symptoms, significant injury)
   - "routine" → general medical information, education, or non‑acute questions
   - "not_applicable" → for non‑medical queries

3. "complexity" (only if query_type = "medical"):
   - "complex" → requires step‑by‑step reasoning, mechanisms, comparisons, differential diagnosis, or multi‑step explanations
     Examples: "How does HIV lead to AIDS?", "Compare type 1 and type 2 diabetes", "Explain how ACE inhibitors work"
   - "simple" → definitions, straightforward facts, yes/no questions, single‑concept explanations
     Examples: "What is AIDS?", "Is there a cure for AIDS?", "What are the symptoms of flu?"
   - "not_applicable" → for non‑medical queries

4. "reasoning":
   - One short sentence explaining why you chose the labels above.

User Query: "{query}"

Respond with **only** a valid JSON object in this exact structure:
{{
  "query_type": "medical|greeting|non_medical",
  "urgency": "emergency|urgent|routine|not_applicable",
  "complexity": "complex|simple|not_applicable",
  "reasoning": "your brief explanation"
}}
""".strip()

    analysis_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
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
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        # Try to find JSON pattern
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        analysis = json.loads(content)
        
        print(f"[LLM Analysis] Query: '{user_message[:50]}...'")
        print(f"[LLM Analysis] Result: {analysis}")
        
        # Additional safety checks
        if analysis["query_type"] == "medical":
            # Force simple complexity for basic definition questions
            basic_question_patterns = [
                r'(what is|what are|define|explain briefly|tell me about)',
                r'(is there a cure|cure for|can you cure|treatment for)',
                r'(symptoms of|signs of|causes of|risk factors for)',
                r'(how to (treat|prevent|diagnose)|prevention of|diagnosis of)',
                r'(yes or no|simple explanation|basic info)'
            ]
            
            query_lower = user_message.lower()
            is_basic_question = any(
                re.search(pattern, query_lower) 
                for pattern in basic_question_patterns
            )
            
            if is_basic_question and analysis["complexity"] == "complex":
                print(f"[LLM Analysis] Overriding: Basic question detected, setting complexity to simple")
                analysis["complexity"] = "simple"
                analysis["reasoning"] = "Basic fact/definition question requiring straightforward answer"
        
        return analysis
        
    except Exception as e:
        print(f"[LLM Analysis Error]: {str(e)}")
        print(f"[LLM Analysis] Raw response: {response.content if 'response' in locals() else 'No response'}")
        # Fallback to simple for most queries
        return {
            "query_type": "medical",
            "urgency": "routine",
            "complexity": "simple",
            "reasoning": "Defaulted due to analysis error - assuming simple medical query"
        }

# ============================================================================
# IMPROVED DYNAMIC PROMPTS WITH BETTER FORMATTING
# ============================================================================

# For COMPLEX queries requiring step-by-step reasoning
complex_medical_prompt = """
You are an expert **Medical AI Assistant** that explains medical topics in depth using clear, evidence‑informed reasoning. 
You are providing **general educational information only**, not diagnosis or treatment.

Formatting rules:
- Use **bold** for key medical terms and section titles.
- Use short paragraphs and bullet lists for readability.
- Organize your answer into logical sections with clear headings.
- Keep the language accessible to an educated layperson.

Use structured, step‑by‑step reasoning internally, but do **not** show chain‑of‑thought explicitly. 
Instead, present a clear, well‑organized explanation.

Recommended answer structure (adapt as needed for the question):

**Direct Answer**
- 1–3 sentences that directly answer the user’s main question.

**Overview**
- Briefly define or describe the main condition, concept, or problem.

**Detailed Explanation**
- Explain important mechanisms, processes, or comparisons in a logical order.
- Clarify how different factors are related.
- When helpful, break complex ideas into short bullet points.

**Key Points**
- Summarize the most important takeaways in 3–5 bullets.

**When to Seek Medical Help**
- Highlight red‑flag symptoms or situations where someone should contact a doctor or emergency services.

Use the retrieved context from the medical database to stay **consistent with evidence‑based guidelines** when possible. Do not invent specific statistics or guideline names if they are not present in the context.

Retrieved Context:
{context}

Conversation History:
{chat_history}

User Question:
{input}
""".strip()


# For SIMPLE queries - PROFESSIONAL FORMATTING
simple_medical_prompt = """
You are a helpful **Medical Information Assistant**. 
Provide clear, concise, and medically sound educational information only (no diagnosis or treatment decisions).

Formatting rules (strict):
- Use **bold** for section titles and important medical terms.
- Use "- " for all bullet points (no "*", "+", or numbered lists unless the user explicitly asks for steps).
- Keep sentences short and direct.
- Avoid revealing your internal reasoning or chain‑of‑thought.

Use this structure unless the question clearly requires a different format:

**Direct Answer**
- 1–2 sentences that directly answer the user’s question.

**Key Information**
- Bullet points with core facts the user should know.
- Define important terms in simple language.

**Symptoms and Causes** (only if relevant)
- Brief bullets for typical **symptoms**.
- Brief bullets for common **causes** or risk factors.

**Important Considerations**
- When to consult a doctor or specialist.
- General treatment approaches in neutral, non‑prescriptive terms.
- Any major warnings or limitations.

Use the retrieved context from the medical database to keep information aligned with current medical knowledge. Do not fabricate details that are not supported by the context.

Retrieved Context:
{context}

Conversation History:
{chat_history}

User Question:
{input}
""".strip()


# For EMERGENCY queries
emergency_medical_prompt = """You are a Medical Emergency Response Assistant.

**URGENT RESPONSE FORMAT**:

**🚨 IMMEDIATE ACTION REQUIRED**
[Clear numbered steps with bold critical actions]

**⚠️ CRITICAL WARNING SIGNS**
[List with bold important symptoms]

**🏥 SEEK MEDICAL CARE NOW**
[Specific instructions with bold locations/actions]

**DO NOT DELAY** - Call emergency services immediately for:
- **Chest pain or pressure**
- **Difficulty breathing**
- **Severe bleeding**
- **Sudden weakness or numbness**

**Retrieved Medical Context:**
{context}

**User Query: {input}**

Provide immediate, actionable emergency guidance with bold critical information."""

# For GREETINGS and CASUAL responses
def generate_greeting_response(user_message: str) -> str:
    """Generate contextual greeting responses"""
    message_lower = user_message.lower().strip()
    
    # Extract name if introduced
    name_match = re.search(r'(?:my name is|i am|im|i\'m)\s+(\w+)', message_lower, re.IGNORECASE)
    
    if name_match:
        name = name_match.group(1).capitalize()
        return f"<strong>Nice to meet you, {name}!</strong> 😊 I'm your Medical AI Assistant. How can I help you with medical information today?"
    
    if any(greeting in message_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "<strong>Hello! 👋</strong> I'm your <strong>Medical AI Assistant</strong>. I can help you understand medical conditions, symptoms, and treatments. What would you like to know?"
    
    if any(thanks in message_lower for thanks in ['thank', 'thanks', 'appreciate']):
        return "<strong>You're welcome! 😊</strong> Feel free to ask if you have any other medical questions. <strong>Stay healthy!</strong>"
    
    if any(bye in message_lower for bye in ['bye', 'goodbye', 'see you']):
        return "<strong>Goodbye!</strong> Take care of your health. Feel free to return anytime you have medical questions! 👋"
    
    # Generic casual response
    return "I'm here to help! 😊 Do you have any <strong>medical questions</strong> or <strong>health concerns</strong> I can assist with?"

# ============================================================================
# INITIALIZE LLMs AND CHAINS
# ============================================================================

# Complex query LLM
complex_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=800,
    timeout=30,
    max_retries=2,
)

# Simple query LLM - with lower temperature for more direct answers
simple_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=600,
    timeout=25,
    max_retries=2,
)

# Emergency LLM
emergency_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=500,
    timeout=30,
    max_retries=2,
)

# Load embeddings and vector store
embeddings = download_embeddings()
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Retriever with adjusted parameters
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
        
        # STEP 1: LLM-DRIVEN QUERY ANALYSIS
        analysis = analyze_query_with_llm(user_message)
        
        query_type = analysis.get("query_type", "medical")
        urgency = analysis.get("urgency", "routine")
        complexity = analysis.get("complexity", "simple")
        
        print(f"[Query Processing] Type: {query_type}, Urgency: {urgency}, Complexity: {complexity}")
        
        # STEP 2: HANDLE NON-MEDICAL AND GREETINGS
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
                "<strong>I'm a specialized medical AI assistant</strong> focused on <strong>health, medical conditions, "
                "symptoms, and treatments</strong>. 🏥<br><br>"
                "<strong>I can help with:</strong><br>"
                "• <strong>Medical conditions</strong> and diseases<br>"
                "• <strong>Symptoms</strong> and their significance<br>"
                "• <strong>Treatment options</strong> and medications<br>"
                "• <strong>Health-related</strong> questions<br><br>"
                "Please ask a <strong>medical or health-related</strong> question! 😊"
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
        
        # STEP 3: HANDLE MEDICAL QUERIES WITH DYNAMIC RESPONSE FORMAT
        if urgency == "emergency":
            print(f"[Processing] EMERGENCY query")
            chain = create_conversational_chain(emergency_medical_prompt, emergency_llm)
            uses_cot = True
            
        elif complexity == "complex":
            print(f"[Processing] COMPLEX query with structured reasoning")
            chain = create_conversational_chain(complex_medical_prompt, complex_llm)
            uses_cot = True
            
        else:  # simple routine queries
            print(f"[Processing] SIMPLE query with professional formatting")
            chain = create_conversational_chain(simple_medical_prompt, simple_llm)
            uses_cot = False
        
        # Invoke the selected chain
        response = chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response["answer"]
        context_docs = response.get("context", [])
        
        # Format response with improved formatting
        formatted_answer = format_response(
            answer,
            uses_cot,
            is_emergency=(urgency == "emergency")
        )
        
        # Add disclaimer if not present and not emergency
        if urgency != "emergency" and "consult" not in formatted_answer.lower():
            disclaimer = '''
            <div class="disclaimer-box">
                <strong>⚠️ Medical Disclaimer:</strong> This information is for educational purposes only. 
                Always consult qualified healthcare professionals for diagnosis and treatment.
            </div>
            '''
            formatted_answer += disclaimer
        
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

def format_response(text: str, is_cot_response: bool = False, is_emergency: bool = False) -> str:
    """Format response text with professional HTML formatting"""

    # --- 1) Light markdown to HTML ---
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic (single *)
    text = re.sub(r'\*(?!\*)(.+?)\*(?!\*)', r'<em>\1</em>', text)

    # --- 2) Convert to lines and build simple lists ---
    lines = text.split("\n")
    html_parts = []
    in_list = False

    section_title_pattern = re.compile(r'^([A-Z][A-Za-z ]+):$')

    for line in lines:
        stripped = line.strip()

        # Section title (e.g., "Direct Answer:")
        if section_title_pattern.match(stripped):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            title = stripped[:-1]  # remove colon
            html_parts.append(f'<div class="section-title"><strong>{title}</strong></div>')
            continue

        # Bullet list item: -, *, +
        if stripped.startswith(("-", "*", "+")) and len(stripped) > 1 and stripped[1] == " ":
            if not in_list:
                html_parts.append('<ul class="bullet-list">')
                in_list = True
            item_text = stripped[2:].strip()
            html_parts.append(f"<li>{item_text}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(stripped + "<br>")

    if in_list:
        html_parts.append("</ul>")

    text = "".join(html_parts)

    # --- 3) Wrap in containers (same visual chrome) ---
    if is_cot_response:
        wrapper = f"""
        <div class="structured-response">
            <div class="response-header">
                <i class="fas fa-brain"></i> Detailed Medical Analysis
            </div>
            <div class="response-content">
                {text}
            </div>
        </div>
        """
    else:
        wrapper = f"""
        <div class="simple-response">
            <div class="response-header">
                <i class="fas fa-stethoscope"></i> Medical Information
            </div>
            <div class="response-content">
                {text}
            </div>
        </div>
        """

    return wrapper


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
