from flask import Flask, request, jsonify, render_template
import json
import os
import re # For keyword extraction and ID matching
# Updated OpenAI import for v1.x+
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError

# --- Requests and BeautifulSoup Imports ---
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
GRANT_DATA_PATH = os.path.join("scripts", "grant_data", "independent_school_district_grants_search_combined.json")
# --- Path for Simulated Financial Data ---
FINANCIAL_DATA_PATH = os.path.join("data", "simulated_financial_data.json") # Assuming it's in a 'data' subfolder

MAX_CONTEXT_GRANTS = 5 # Limit grants selected by keyword
MAX_HISTORY_LENGTH = 10 # Limit history turns sent to LLM
MAX_FETCHED_CONTENT_LENGTH = 4000 # Limit characters sent from fetched page
REQUESTS_TIMEOUT = 15 # Seconds to wait for HTTP request

# --- OpenAI API Setup ---
try:
    client = OpenAI()
except Exception as e:
     print(f"Error initializing OpenAI client: {e}. OpenAI calls will likely fail.")
     client = None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Variables for Data ---
grant_data = []
grant_data_by_id = {}
# --- Load Simulated Financial Data ---
simulated_financial_data = None
try:
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory where app.py is located

    # Load Grant Data
    absolute_grant_data_path = os.path.join(base_dir, GRANT_DATA_PATH)
    print(f"Attempting to load grant data from: {absolute_grant_data_path}")
    with open(absolute_grant_data_path, 'r', encoding='utf-8') as f:
        grant_data = json.load(f)
        for grant_item in grant_data: # Renamed to avoid conflict with module name
            if grant_item.get("opportunityID"):
                 grant_data_by_id[grant_item["opportunityID"]] = grant_item
    print(f"Successfully loaded {len(grant_data)} grant records.")

    # Load Simulated Financial Data
    absolute_financial_data_path = os.path.join(base_dir, FINANCIAL_DATA_PATH)
    print(f"Attempting to load simulated financial data from: {absolute_financial_data_path}")
    with open(absolute_financial_data_path, 'r', encoding='utf-8') as f:
        simulated_financial_data = json.load(f)
    print("Successfully loaded simulated financial data.")

except FileNotFoundError as e:
    # Check which file was not found by comparing the error message with the paths
    if GRANT_DATA_PATH in str(e) or absolute_grant_data_path in str(e): # Check both relative and absolute
        print(f"Error: Grant data file not found. Path checked: {absolute_grant_data_path}. Grant context will be limited.")
    elif FINANCIAL_DATA_PATH in str(e) or absolute_financial_data_path in str(e): # Check both relative and absolute
        print(f"Error: Simulated financial data file not found. Path checked: {absolute_financial_data_path}. Financial context will be missing.")
        # Allow app to run without financial data, but it won't be used
    else:
        print(f"Error: A data file was not found: {e}")
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from a data file. Check file integrity. Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred loading data: {e}")


# --- Requests + BeautifulSoup Fetch Function ---
def fetch_with_requests_bs4(url: str) -> str:
    """Fetches page content using requests and parses with BeautifulSoup."""
    print(f"--- Attempting to fetch with Requests+BS4: {url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }
    try:
        response = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Check if content type is HTML before parsing
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Warning: Content type is not HTML ({content_type}) for {url}. Returning raw text.")
            return response.text if response.text else "[No text content returned]"
        soup = BeautifulSoup(response.content, 'html.parser')
        # Try finding a main content area first, otherwise fallback to body
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content') or soup.find('div', class_='content')
        if main_content:
            content = main_content.get_text(separator='\n', strip=True)
            print("Extracted text from main content area.")
        else:
            # Fallback to getting text from the whole body
            body = soup.find('body')
            content = body.get_text(separator='\n', strip=True) if body else "[Could not find body tag]"
            print("Extracted text from body (fallback).")
        print(f"Requests+BS4 successfully fetched content (first 100 chars): {content[:100]}")
        return content
    except requests.exceptions.Timeout:
        print(f"Requests TimeoutException for {url}")
        return f"[Error fetching content with Requests: Timeout connecting to {url}]"
    except requests.exceptions.RequestException as e:
        print(f"Requests RequestException for {url}: {e}")
        return f"[Error fetching content with Requests: {e}]"
    except Exception as e:
        print(f"Unexpected error during Requests+BS4 fetch for {url}: {e}")
        return f"[Unexpected error fetching content with Requests for {url}]"


# --- Helper Functions for Context and ID ---
def extract_keywords(text):
    if not text: return set()
    words = re.findall(r'\b\w+\b', text.lower())
    # Updated stop_words to include financial terms if they are too generic for keyword search
    stop_words = {"a", "an", "the", "is", "are", "in", "on", "for", "of", "and", "to", "what", "who", "tell", "me", "about", "grant", "grants", "more", "details", "detail", "help", "write", "financial", "budget", "funding"}
    return set(word for word in words if word not in stop_words and len(word) > 2)

def find_opportunity_id(text, grant_lookup):
    potential_ids = re.findall(r'\b(\d{6,7})\b', text) # Assumes Opportunity IDs are 6-7 digits
    for pid in potential_ids:
        if pid in grant_lookup:
            print(f"Found potential Opportunity ID in question: {pid}")
            return pid
    return None

def select_relevant_grants_by_keyword(question, all_grants_list, max_grants=MAX_CONTEXT_GRANTS): # Renamed all_grants to all_grants_list
    keywords = extract_keywords(question)
    if not keywords: return []
    print(f"Keywords for search: {keywords}")
    relevant_grants = []
    for grant_item in all_grants_list: # Use the new parameter name
        search_text = f"{grant_item.get('opportunityTitle', '')} {grant_item.get('description', '')} {grant_item.get('opportunityCategory', '')}".lower()
        if any(keyword in search_text for keyword in keywords):
            relevant_grants.append(grant_item)
            if len(relevant_grants) >= max_grants: break
    print(f"Found {len(relevant_grants)} relevant grants via keywords.")
    return relevant_grants

# --- Helper Function for OpenAI Interaction (Updated System Prompt & Context) ---
def get_openai_response(full_history, json_grant_context_str, fetched_web_content_status=None, internal_financial_context_str=None):
    """
    Sends history and various contexts to OpenAI and returns the response.
    """
    print("--- Preparing OpenAI Request ---")

    if not client:
         print("Error: OpenAI client not initialized.")
         return "Sorry, the chatbot is not configured correctly (OpenAI client issue)."

    # --- UPDATED SYSTEM PROMPT ---
    system_prompt = (
        "You are a helpful AI assistant for Springfield Independent School District, specializing in grant information. "
        "Your primary goal is to answer questions about specific grants based *only* on the provided context: "
        "the 'JSON Grant Data Context' (external grant opportunities) and the 'Fetched Webpage Content Status' (live details of a specific external grant). "
        "You also have access to 'Internal School District Financial Context'.\n\n"
        "RULES FOR ANSWERING ABOUT SPECIFIC EXTERNAL GRANTS:\n"
        "* Base answers *strictly* on the provided JSON grant data and fetched web content.\n"
        "* If 'Fetched Webpage Content Status' contains actual content, prioritize it for specific details about one grant.\n"
        "* If 'Fetched Webpage Content Status' indicates an error or no content, inform the user you couldn't retrieve live details and rely *only* on the JSON grant data and history.\n"
        "* If the information needed is not present in the provided context or history, clearly state that.\n"
        "* Do not make up information or use external knowledge for external grant-specific questions.\n"
        "* Refer to specific grants by title or ID when possible. Provide the grant link if relevant.\n\n"
        "USING INTERNAL SCHOOL DISTRICT FINANCIAL CONTEXT:\n"
        "* If the user's question relates to the school district's own finances, budget, needs, or how a potential grant aligns with these, "
        "use the 'Internal School District Financial Context' to inform your answer. "
        "For example, if asked about the district's budget for technology or what the district needs funding for.\n"
        "* When discussing applying for a grant, you can use the internal financial context to explain why the district needs the grant or how it fits into existing priorities/budget.\n\n"
        "EXCEPTION - GENERAL GRANT WRITING HELP:\n"
        "* If the user explicitly asks for general help or tips on *writing* a grant, you MAY provide general advice. "
        "This advice should be clearly marked as general. Do NOT offer to write the grant *for* the user.\n"
        "* Keep grant writing advice concise (e.g., understanding requirements, clear objectives, budget planning, proofreading).\n\n"
        "Always be concise and helpful. If context is missing for any part of a question, state that clearly."
    )
    # --- END UPDATED SYSTEM PROMPT ---

    messages_for_api = [{"role": "system", "content": system_prompt}]
    limited_history = full_history[-(MAX_HISTORY_LENGTH * 2):-1] # Get all but the last message
    messages_for_api.extend(limited_history)

    latest_user_message = full_history[-1] # The last message is the current user question
    if latest_user_message['role'] != 'user':
        print("Error: Last message in history is not from the user.")
        return "Sorry, there was an internal error processing the conversation history."

    # Construct the user message content, including all relevant contexts
    user_content_parts = [f"User Question:\n{latest_user_message['content']}"]
    if json_grant_context_str:
         user_content_parts.append(f"\n\nJSON Grant Data Context (External Opportunities):\n```json\n{json_grant_context_str}\n```")
    if fetched_web_content_status:
         user_content_parts.append(f"\n\nFetched Webpage Content Status (External Opportunity Detail):\n```\n{fetched_web_content_status}\n```")
    # --- Add Internal Financial Context if available ---
    if internal_financial_context_str:
        user_content_parts.append(f"\n\nInternal School District Financial Context:\n```json\n{internal_financial_context_str}\n```")


    latest_user_message_with_context = "".join(user_content_parts)
    messages_for_api.append({"role": "user", "content": latest_user_message_with_context})

    try:
        print(f"Sending request to OpenAI API (v1.x+) with {len(messages_for_api)} messages...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_api,
            temperature=0.5,
            max_tokens=800, # Increased slightly for potentially more complex answers
        )
        assistant_response = response.choices[0].message.content.strip()
        print(f"OpenAI Response Received: {assistant_response[:100]}...")
        return assistant_response
    # (Error handling remains the same)
    except AuthenticationError as e:
         print(f"OpenAI Authentication Error: {e}")
         error_detail = f" ({e.body.get('message', '')})" if hasattr(e, 'body') and isinstance(e.body, dict) else ""
         return f"Sorry, there's an issue with the chatbot configuration (Authentication Error{error_detail}). Please check the API key."
    except RateLimitError as e:
         print(f"OpenAI Rate Limit Error: {e}")
         return "Sorry, the chatbot is currently experiencing high traffic (Rate Limit Exceeded). Please try again later."
    except OpenAIError as e:
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


# --- API Endpoint for Chat (Uses Requests+BS4 for Fetching) ---
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

        context_grants_json_str = "" # For external grants
        fetched_content_status = None
        internal_financial_context_str = "" # For school's own financial data

        target_grant_id = None
        # Simple check for grant writing help, or questions about district's own finances
        is_general_request = ("write" in user_question.lower() and "grant" in user_question.lower()) or \
                             ("budget" in user_question.lower()) or \
                             ("financial" in user_question.lower() and "district" in user_question.lower()) or \
                             ("funding need" in user_question.lower())

        # --- Determine Context Strategy ---
        if not is_general_request: # If specific to external grants
            target_grant_id = find_opportunity_id(user_question, grant_data_by_id)
            selected_grants_for_context = []

            if target_grant_id:
                print(f"Specific grant ID {target_grant_id} identified. Attempting to fetch details.")
                specific_grant = grant_data_by_id.get(target_grant_id)
                if specific_grant:
                    selected_grants_for_context = [specific_grant]
                    grant_link = specific_grant.get("link")
                    if grant_link:
                        fetched_content_raw = fetch_with_requests_bs4(grant_link)
                        print(f"Raw content returned by Requests+BS4: {fetched_content_raw[:100] if fetched_content_raw else 'None'}...")
                        if fetched_content_raw and not fetched_content_raw.startswith("[Error") and not fetched_content_raw.startswith("[No content") and not fetched_content_raw.startswith("[Could not find"):
                             fetched_content_status = str(fetched_content_raw)[:MAX_FETCHED_CONTENT_LENGTH]
                             print(f"Fetched content stored (truncated to {MAX_FETCHED_CONTENT_LENGTH} chars).")
                        elif fetched_content_raw: # It's an error/status message from fetch_with_requests_bs4
                             print(f"Requests+BS4 fetch failed or returned no content: {fetched_content_raw}")
                             fetched_content_status = fetched_content_raw
                        else: # Should ideally not happen if fetch function returns error strings
                             print("Requests+BS4 fetch returned None or empty string unexpectedly.")
                             fetched_content_status = "[No content returned from fetch attempt.]"
                    else: # No link found for the specific grant
                        print(f"No link found for grant ID {target_grant_id}.")
                        # selected_grants_for_context is already [specific_grant] from JSON
                else: # Grant ID mentioned but not found in our loaded data
                     print(f"Grant ID {target_grant_id} mentioned but not found in loaded data.")
                     # Fallback to keyword search if ID not found
                     selected_grants_for_context = select_relevant_grants_by_keyword(user_question, grant_data) # Pass the global grant_data
            else: # No specific ID found in question, use keyword search for external grants
                print("No specific grant ID found in question, using keyword search for context.")
                selected_grants_for_context = select_relevant_grants_by_keyword(user_question, grant_data) # Pass the global grant_data
            
            if selected_grants_for_context:
                context_grants_json_str = json.dumps(selected_grants_for_context, indent=2)

        else: # General request (writing help or about district finances)
             print("General request detected. Context will primarily be internal financial data if relevant.")
             # For general writing help, no specific external grant context is needed.
             # For questions about district finances, only internal financial data is primary.

        # --- Always include Internal Financial Context if available ---
        if simulated_financial_data:
            internal_financial_context_str = json.dumps(simulated_financial_data, indent=2)


        # --- Append current question to history ---
        current_turn_history = conversation_history + [{"role": "user", "content": user_question}]

        # --- Get Response ---
        bot_response = get_openai_response(
            current_turn_history,
            context_grants_json_str, # External grant JSON
            fetched_content_status,    # Fetched web content for a specific external grant
            internal_financial_context_str # Internal financial data
        )

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
