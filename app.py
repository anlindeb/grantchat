from flask import Flask, request, jsonify, render_template
import json
import os
import re # For keyword extraction and ID matching
# Updated OpenAI import for v1.x+
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError
# NOTE: No need to import browsing tool directly when using tool_code

# --- Configuration ---
GRANT_DATA_PATH = os.path.join("scripts", "grant_data", "independent_school_district_grants_search_combined.json")
MAX_CONTEXT_GRANTS = 5 # Limit grants selected by keyword
MAX_HISTORY_LENGTH = 10 # Limit history turns sent to LLM
MAX_BROWSED_CONTENT_LENGTH = 3500 # Limit characters sent from browsed page (increased slightly)

# --- OpenAI API Setup ---
try:
    client = OpenAI()
except Exception as e:
     print(f"Error initializing OpenAI client: {e}. OpenAI calls will likely fail.")
     client = None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Variable for Grant Data ---
grant_data = []
grant_data_by_id = {} # Add a dictionary for quick ID lookups
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_grant_data_path = os.path.join(base_dir, GRANT_DATA_PATH)
    print(f"Attempting to load grant data from: {absolute_grant_data_path}")
    with open(absolute_grant_data_path, 'r', encoding='utf-8') as f:
        grant_data = json.load(f)
        # Populate the lookup dictionary
        for grant in grant_data:
            if grant.get("opportunityID"):
                 grant_data_by_id[grant["opportunityID"]] = grant
    print(f"Successfully loaded {len(grant_data)} grant records.")
except FileNotFoundError:
    print(f"Error: Grant data file not found at {absolute_grant_data_path}. Chatbot may not have grant context.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {absolute_grant_data_path}. Check file integrity.")
except Exception as e:
    print(f"An unexpected error occurred loading grant data: {e}")

# --- Simple Keyword Extraction Helper ---
def extract_keywords(text):
    """Extracts simple keywords from text (lowercase, alphanumeric)."""
    if not text:
        return set()
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {"a", "an", "the", "is", "are", "in", "on", "for", "of", "and", "to", "what", "who", "tell", "me", "about", "grant", "grants", "more", "details", "detail"}
    return set(word for word in words if word not in stop_words and len(word) > 2)

# --- Grant ID/Number Extraction Helper ---
def find_opportunity_id(text, grant_lookup):
    """Attempts to find a valid Opportunity ID mentioned in the text."""
    potential_ids = re.findall(r'\b(\d{6,7})\b', text)
    for pid in potential_ids:
        if pid in grant_lookup:
            print(f"Found potential Opportunity ID in question: {pid}")
            return pid
    return None

# --- Context Selection Function (Keywords) ---
def select_relevant_grants_by_keyword(question, all_grants, max_grants=MAX_CONTEXT_GRANTS):
    """Selects grants relevant to the question based on keywords."""
    keywords = extract_keywords(question)
    if not keywords:
        print("No keywords extracted for keyword search.")
        return []

    print(f"Keywords for search: {keywords}")
    relevant_grants = []
    for grant in all_grants:
        search_text = f"{grant.get('opportunityTitle', '')} {grant.get('description', '')} {grant.get('opportunityCategory', '')}".lower()
        if any(keyword in search_text for keyword in keywords):
            relevant_grants.append(grant)
            if len(relevant_grants) >= max_grants:
                break

    print(f"Found {len(relevant_grants)} relevant grants via keywords.")
    return relevant_grants

# --- Helper Function for OpenAI Interaction (Handles Fetched Content) ---
def get_openai_response(full_history, context_str, fetched_content_str=None):
    """
    Sends history, context (JSON grants), and optional fetched web content
    to OpenAI (v1.x+) and returns the response.
    """
    print("--- Preparing OpenAI Request ---")

    if not client:
         print("Error: OpenAI client not initialized.")
         return "Sorry, the chatbot is not configured correctly (OpenAI client issue)."

    system_prompt = (
        "You are a helpful assistant specializing in grant information for Independent School Districts. "
        "Your knowledge is based *only* on the grant data provided in the user's message context (JSON Grant Data Context) "
        "and potentially additional details fetched from the grant's webpage (Fetched Webpage Content). "
        "Answer the user's question concisely based *only* on the provided information. "
        "Prioritize information from the 'Fetched Webpage Content' if available and relevant for specific details about one grant. "
        "Use the JSON Grant Data Context for summaries or comparing multiple grants. "
        "Use the conversation history for context about previous turns. "
        "If the information needed is not present, clearly state that. "
        "Do not make up information or use external knowledge. Refer to specific grants by title or ID when possible. Provide the grant link if relevant."
        "If asked generally about a grant mentioned in the previous turn, use the context provided in that turn."
    )

    messages_for_api = [{"role": "system", "content": system_prompt}]
    limited_history = full_history[-(MAX_HISTORY_LENGTH * 2):-1]
    messages_for_api.extend(limited_history)

    latest_user_message = full_history[-1]
    if latest_user_message['role'] != 'user':
        print("Error: Last message in history is not from the user.")
        return "Sorry, there was an internal error processing the conversation history."

    # Construct the user message content, including fetched content if available
    user_content_parts = [f"User Question:\n{latest_user_message['content']}"]
    if context_str:
         user_content_parts.append(f"\n\nJSON Grant Data Context:\n```json\n{context_str}\n```")
    # Add fetched content only if it's not None or an error message
    if fetched_content_str and not fetched_content_str.startswith("[Error") and not fetched_content_str.startswith("[No content"):
         user_content_parts.append(f"\n\nFetched Webpage Content:\n```\n{fetched_content_str}\n```")
    elif fetched_content_str: # If it's an error or no content message, log it but don't send to LLM
         print(f"Note: Not sending fetched content to LLM: {fetched_content_str}")


    latest_user_message_with_context = "".join(user_content_parts)
    messages_for_api.append({"role": "user", "content": latest_user_message_with_context})

    try:
        print(f"Sending request to OpenAI API (v1.x+) with {len(messages_for_api)} messages...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_api,
            temperature=0.3,
            max_tokens=700,
        )
        assistant_response = response.choices[0].message.content.strip()
        print(f"OpenAI Response Received: {assistant_response[:100]}...")
        return assistant_response
    # (Error handling remains the same as previous version)
    except AuthenticationError as e:
         print(f"OpenAI Authentication Error: {e}")
         error_detail = f" ({e.body.get('message', '')})" if hasattr(e, 'body') and isinstance(e.body, dict) else ""
         return f"Sorry, there's an issue with the chatbot configuration (Authentication Error{error_detail}). Please check the API key."
    except RateLimitError as e:
         print(f"OpenAI Rate Limit Error: {e}")
         return "Sorry, the chatbot is currently experiencing high traffic (Rate Limit Exceeded). Please try again later."
    except OpenAIError as e: # Catch other OpenAI specific errors
        error_type = type(e).__name__
        if "context_length_exceeded" in str(e).lower():
             print(f"OpenAI API Error ({error_type}): Context length exceeded. {e}")
             return "Sorry, the conversation history or the provided grant data is too long for the AI model to process. Please try starting a new topic or asking a more specific question."
        print(f"OpenAI API Error ({error_type}): {e}")
        return f"Sorry, I encountered an error trying to reach the AI model ({error_type})."
    except Exception as e:
        error_type = type(e).__name__
        print(f"Generic Error calling OpenAI API ({error_type}): {e}")
        return f"Sorry, I encountered an unexpected error ({error_type}) while processing your request."


# --- API Endpoint for Chat (Handles Context Selection & Fetching) ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat messages, selects context, fetches web content if needed."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        user_question = data['question']
        conversation_history = data.get('history', [])
        print(f"Received question: {user_question}")

        context_grants = []
        fetched_content_raw = None # Initialize raw fetched content
        fetched_content = None # Initialize truncated fetched content
        target_grant_id = None

        # --- Determine Context Strategy ---
        target_grant_id = find_opportunity_id(user_question, grant_data_by_id)

        if target_grant_id:
            print(f"Specific grant ID {target_grant_id} identified. Attempting to fetch details.")
            specific_grant = grant_data_by_id.get(target_grant_id)
            if specific_grant:
                context_grants = [specific_grant]
                grant_link = specific_grant.get("link")
                if grant_link:
                    print(f"Attempting to browse URL: {grant_link}")
                    try:
                        # --- !!! Use browsing tool via tool_code !!! ---
                        # @@tool_code
                        # print(f"Calling browsing tool for URL: {grant_link}")
                        # fetched_content_raw = browsing.browse(url=grant_link, query=user_question)
                        # print(f"Browsing tool returned content (type: {type(fetched_content_raw)}, first 100 chars): {str(fetched_content_raw)[:100] if fetched_content_raw else 'None'}") # Added type logging
                        # @@
                        # --- End tool_code block ---

                        # --- Add explicit logging of the raw result ---
                        print(f"Raw content returned by tool_code: {fetched_content_raw!r}") # Use !r for unambiguous representation (shows None, empty strings clearly)

                        if fetched_content_raw: # Check if it's not None and not empty string
                             fetched_content = str(fetched_content_raw)[:MAX_BROWSED_CONTENT_LENGTH] # Ensure string conversion
                             print(f"Fetched content stored (truncated to {MAX_BROWSED_CONTENT_LENGTH} chars).")
                        else:
                             print("Browsing tool did not return content (returned None or empty string).")
                             fetched_content = "[No content returned from browsing tool]" # Keep error message

                    except Exception as e:
                        print(f"Error during browsing tool execution for {grant_link}: {e}")
                        fetched_content = f"[Error fetching content from {grant_link}]"
                else:
                    print(f"No link found for grant ID {target_grant_id}.")
                    context_grants = select_relevant_grants_by_keyword(user_question, grant_data)
            else:
                 print(f"Grant ID {target_grant_id} mentioned but not found in loaded data.")
                 context_grants = select_relevant_grants_by_keyword(user_question, grant_data)
        else:
            print("No specific grant ID found in question, using keyword search for context.")
            context_grants = select_relevant_grants_by_keyword(user_question, grant_data)


        # --- Prepare Context Strings ---
        context_str = json.dumps(context_grants, indent=2) if context_grants else ""
        fetched_content_str = fetched_content if fetched_content else None

        # --- Append current question to history ---
        current_turn_history = conversation_history + [{"role": "user", "content": user_question}]

        # --- Get Response ---
        bot_response = get_openai_response(current_turn_history, context_str, fetched_content_str)

        return jsonify({"answer": bot_response})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        # import traceback
        # traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Route for the HTML Frontend ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
