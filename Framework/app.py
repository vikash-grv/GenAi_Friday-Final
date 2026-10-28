import os
import re
import atexit
import signal
import sys
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response, Response, stream_with_context
from flask import Blueprint
from jinja2 import ChoiceLoader, FileSystemLoader
from flasgger import Swagger
from langchain_core.messages import SystemMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_cls
from stores import chat_store, knowledge_store, plans_store, approvals_store, memory_store
from core import attachments, hitl, usage, events, memory, mcp_client
from core.paths import ROOT
from core.domain import CONFIG, ID, TYPES, UI, STATIC_DIR, TEMPLATES_DIR, adapter, text
from core.prompts import PROMPT_SECTIONS
from core.conversation import handle_message, dashboard_view, compare_agents_with_web, HISTORY_MESSAGES
from core.llm import chat_llm
from core.auth import (
    register_user, authenticate_user, refresh_access_token, logout_user,
    token_required, get_token_from_header, is_token_blacklisted, decode_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.jinja_loader = ChoiceLoader([FileSystemLoader(TEMPLATES_DIR), app.jinja_loader])
app.register_blueprint(Blueprint('domain', __name__, static_folder=STATIC_DIR, static_url_path='/domain-static'))


@app.context_processor
def inject_domain():
    roles = {trigger: role for role, trigger in CONFIG['hitl_triggers'].items()}
    return {'ui': UI, 'domain_ui': {**UI, 'types': TYPES, 'id_field': ID, 'approval_roles': roles}}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": CONFIG["app"]["api_title"],
        "description": CONFIG["app"]["api_description"],
        "version": "1.0.0",
        "contact": {
            "name": CONFIG["app"]["api_contact"]
        }
    },
    "host": "127.0.0.1:5000",
    "basePath": "/",
    "schemes": ["http"],
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "JWT Authorization header using the Bearer scheme. Example: 'Bearer {token}'"
        }
    },
    "security": [{"Bearer": []}]
}

Swagger(app, config=swagger_config, template=swagger_template)

ACCESS_COOKIE = 'access_token'
REFRESH_COOKIE = 'refresh_token'

log = logging.getLogger(__name__)

MAX_CHAT_FILES = 5

CHAT_ID_RE = re.compile(r'^[0-9a-f]{32}$')
RUN_ID_RE = re.compile(r'[0-9a-f]{32}')

def valid_chat_id(chat_id):
    return bool(chat_id and CHAT_ID_RE.match(chat_id))

def set_auth_cookies(response, access_token, refresh_token=None):
    response.set_cookie(ACCESS_COOKIE, access_token, httponly=True, samesite='Lax',
                        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    if refresh_token:
        response.set_cookie(REFRESH_COOKIE, refresh_token, httponly=True, samesite='Lax',
                            max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400)
    return response

def clear_auth_cookies(response):
    response.delete_cookie(ACCESS_COOKIE)
    response.delete_cookie(REFRESH_COOKIE)
    return response

def is_valid_access_token(token):
    if not token or is_token_blacklisted(token):
        return False
    payload = decode_token(token)
    return 'error' not in payload and payload.get('type') == 'access'

def get_cookie_session():
    if is_valid_access_token(request.cookies.get(ACCESS_COOKIE)):
        return True, None
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token:
        result = refresh_access_token(refresh_token)
        if result['success']:
            return True, result['access_token']
    return False, None

def auth_success_response(auth_result):
    response = jsonify({
        'success': True,
        'redirect': url_for('chat_page'),
        'access_token': auth_result['access_token'],
        'refresh_token': auth_result['refresh_token']
    })
    return set_auth_cookies(response, auth_result['access_token'], auth_result['refresh_token'])

@app.route('/')
def index():
    valid, _ = get_cookie_session()
    return redirect(url_for('chat_page' if valid else 'login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User Login / Signup
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            username:
              type: string
              example: "secureuser"
            password:
              type: string
              example: "SecurePass123"
            email:
              type: string
              example: "user@example.com"
            action:
              type: string
              enum: ["login", "signup"]
              example: "login"
    responses:
      200:
        description: Login/Signup successful
        schema:
          type: object
          properties:
            success:
              type: boolean
            redirect:
              type: string
            access_token:
              type: string
            refresh_token:
              type: string
      400:
        description: Invalid credentials or validation error
    """
    if request.method == 'GET':
        valid, new_access = get_cookie_session()
        if valid:
            response = redirect(url_for('chat_page'))
            return set_auth_cookies(response, new_access) if new_access else response
        return render_template('login.html')

    data = request.get_json() or request.form
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    action = data.get('action', 'login')

    if action == 'signup':
        if not username or not email or not password:
            return jsonify({'success': False, 'error': 'All fields are required'})
        result = register_user(username, email, password)
        if result['success']:
            auth_result = authenticate_user(username, password)
            if auth_result['success']:
                return auth_success_response(auth_result)
        return jsonify(result)

    else:
        result = authenticate_user(username, password)
        if result['success']:
            return auth_success_response(result)
        return jsonify(result)

@app.route('/refresh', methods=['POST'])
def refresh():
    """
    Refresh Access Token
    ---
    tags:
      - Authentication
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            refresh_token:
              type: string
              example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    responses:
      200:
        description: New access token issued
        schema:
          type: object
          properties:
            success:
              type: boolean
            access_token:
              type: string
      401:
        description: Invalid or revoked refresh token
    """
    data = request.get_json() or {}
    refresh_token = data.get('refresh_token', '').strip() or request.cookies.get(REFRESH_COOKIE, '')

    if not refresh_token:
        return jsonify({'success': False, 'error': 'Refresh token required'}), 400

    result = refresh_access_token(refresh_token)
    if result['success']:
        return set_auth_cookies(jsonify(result), result['access_token'])
    return jsonify(result), 401

@app.route('/logout', methods=['POST', 'GET'])
def logout():
    """
    Logout User (Revoke Tokens)
    ---
    tags:
      - Authentication
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            access_token:
              type: string
            refresh_token:
              type: string
    responses:
      200:
        description: Logged out successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            redirect:
              type: string
    """
    if request.method == 'POST':
        data = request.get_json() or {}
        access_token = data.get('access_token', '').strip()
        refresh_token = data.get('refresh_token', '').strip()
    else:
        access_token = get_token_from_header() or ''
        refresh_token = request.args.get('refresh_token', '')

    access_token = access_token or request.cookies.get(ACCESS_COOKIE, '')
    refresh_token = refresh_token or request.cookies.get(REFRESH_COOKIE, '')

    if access_token:
        logout_user(access_token, refresh_token if refresh_token else None)
    elif refresh_token:
        logout_user(refresh_token)

    if request.method == 'GET':
        return clear_auth_cookies(redirect(url_for('login')))
    return clear_auth_cookies(jsonify({'success': True, 'redirect': url_for('login')}))

def render_protected(template):
    valid, new_access = get_cookie_session()
    if not valid:
        return clear_auth_cookies(redirect(url_for('login')))
    response = make_response(render_template(template))
    response.headers['Cache-Control'] = 'no-store'
    return set_auth_cookies(response, new_access) if new_access else response

@app.route('/chat-page')
def chat_page():
    return render_protected('chat.html')

@app.route('/dashboard')
def dashboard_page():
    return render_protected('dashboard.html')

@app.route('/reports')
def reports_page():
    return render_protected('reports.html')

@app.route('/knowledge')
def knowledge_page():
    return render_protected('knowledge.html')

@app.route('/calendar')
def calendar_page():
    return render_protected('calendar.html')

@app.route('/booking')
def booking_page():
    return render_protected('booking.html')

@app.route('/clearances')
def clearances_page():
    return render_protected('clearances.html')

@app.route('/memory')
def memory_page():
    return render_protected('memory.html')

@app.route('/mcp')
def mcp_page():
    return render_protected('mcp.html')

def persist_turn(chat_id, user_id, username, meta, state, stored_input, result, attachment_meta=None):
    redirect_to = None
    action = result.get('action') or {}
    if action.get('type') == 'book':
        record, plan_text = plan_from_chat(chat_id, user_id, state)
        if record is None:
            result['response'] = text("booking_prepare_failed", error=plan_text)
        else:
            plan, _ = plans_store.save_plan(user_id, chat_id, record, plan_text, status=action['status'])
            if action.get('checkout'):
                redirect_to = f"/booking?plan_id={plan['plan_id']}"

    title = meta.get('title', 'New chat')
    title_source = meta.get('title_source', 'default')
    if result.get('title'):
        title, title_source = result['title'], 'route'
    elif title_source == 'default' and result['intent'] not in ('greeting', 'help', ''):
        title, title_source = stored_input[:60], 'message'

    saved = True
    try:
        chat_store.add_messages(chat_id, user_id, username, [
            {'role': 'user', 'content': stored_input, 'intent': result['intent'], 'attachment': attachment_meta},
            {'role': 'bot', 'content': result['response'], 'type': result['type'],
             'intent': result['intent'], 'suggestions': result['suggestions'],
             'sources': result.get('sources'), 'usage': result.get('usage'),
             'tools': result.get('tools'), 'memory': result.get('memory')},
        ])
        updates = {'state': state, 'title': title, 'title_source': title_source}
        if result.get('route'):
            updates['route'] = result['route']
        chat_store.update_chat(chat_id, user_id,
                               reembed=title != meta.get('title') or bool(result.get('route')),
                               **updates)
    except Exception:
        log.exception('Saving chat %s failed', chat_id)
        saved = False

    return {
        'chat_id': chat_id,
        'response': result['response'],
        'type': result['type'],
        'suggestions': result['suggestions'],
        'title': title,
        'saved': saved,
        'guardrail': result.get('guardrail'),
        'sources': result.get('sources') or [],
        'attachments': attachment_meta or [],
        'dashboard': dashboard_view(state),
        'intent': result.get('intent', ''),
        'approval': result.get('approval'),
        'drafts': result.get('drafts'),
        'redirect': redirect_to,
        'pending_approvals': approvals_store.count_pending(user_id),
        'usage': result.get('usage'),
        'tools': result.get('tools'),
        'memory': result.get('memory'),
    }

@app.route('/chat', methods=['POST'])
@token_required
def chat():
    """
    Chat with the AI assistant (session based)
    ---
    tags:
      - Chat
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Hello, what can you do?"
            chat_id:
              type: string
              description: Existing chat session id. Omit to start a new chat.
      - in: formData
        name: file
        type: file
        required: false
        description: Optional attachments (any format, repeat the field for several files - up to 5)
          sent as multipart/form-data with "message" and "chat_id". Screenshots/documents add request
          details; voice messages (mp3, wav, m4a, webm, ogg...) are transcribed and used as the user's words.
    responses:
      200:
        description: Chat response
        schema:
          type: object
          properties:
            chat_id:
              type: string
            response:
              type: string
            type:
              type: string
              description: text, guardrail, approval, or a domain result type (see domain.json "types")
            suggestions:
              type: array
              items:
                type: string
            title:
              type: string
            saved:
              type: boolean
            approval:
              type: object
              description: Set when the flow is paused for the user (human in the loop, see prompt_md
                HUMAN_IN_THE_LOOP). Answer by sending one of the option labels as the next message.
            drafts:
              type: array
              items:
                type: object
            redirect:
              type: string
              description: Page to open next, e.g. the checkout after the user confirms a booking
      401:
        description: Unauthorized - invalid or expired token
      404:
        description: Chat not found
    """
    data = request.form if request.files else (request.get_json(silent=True) or {})
    user_input = (data.get('message') or '').strip()
    chat_id = (data.get('chat_id') or '').strip()
    user_id = request.current_user['user_id']
    username = request.current_user['username']
    uploads = [f for f in request.files.getlist('file') + request.files.getlist('files') if f and f.filename][:MAX_CHAT_FILES]

    if not user_input and not uploads:
        return jsonify({'response': 'Please enter a message.', 'type': 'text', 'suggestions': []})

    if chat_id:
        meta = chat_store.get_chat_meta(chat_id, user_id) if valid_chat_id(chat_id) else None
        if meta is None:
            return jsonify({'success': False, 'error': 'Chat not found'}), 404
    else:
        chat_id = chat_store.create_chat(user_id, username)
        meta = chat_store.get_chat_meta(chat_id, user_id)

    state = chat_store.get_state(chat_id, user_id)
    history = chat_store.get_messages(chat_id, user_id, limit=HISTORY_MESSAGES)
    analysed, attachment_meta = [], None
    run_id = (data.get('run_id') or '').strip()
    run_id = run_id if RUN_ID_RE.fullmatch(run_id) else None
    with events.run(run_id, user_id), usage.collect() as collector:
        if uploads:
            today = date_cls.today().strftime('%Y-%m-%d (%A)')
            files = [(f.filename or 'attachment', f.read()) for f in uploads]
            with ThreadPoolExecutor(max_workers=len(files)) as pool:
                tasks = [pool.submit(usage.carry(attachments.analyze), name, blob, today) for name, blob in files]
                analysed = [t.result() for t in tasks]
            attachment_meta = [{'filename': a['filename'], 'kind': a['kind'],
                                'label': attachments.short_label(a) if a['kind'] != 'error' else a['error'],
                                'summary': a.get('summary', ''), 'error': a['kind'] == 'error'} for a in analysed]
        approval_id = (data.get('approval_id') or '').strip() or None
        result = handle_message(user_input, history, state, user_id=user_id, attachments_list=analysed,
                                chat_id=chat_id, approval_id=approval_id)
        result['usage'] = collector.summary()
    spoken = ' '.join(a['transcript'] for a in analysed if a.get('kind') == 'audio')
    stored_input = user_input or (f"🎤 {spoken}" if spoken else '') or ('📎 ' + ', '.join(m['filename'] for m in attachment_meta))
    return jsonify(persist_turn(chat_id, user_id, username, meta, state, stored_input, result, attachment_meta))

@app.route('/chats', methods=['GET'])
@token_required
def list_chats():
    """
    List the current user's chat sessions (newest first)
    ---
    tags:
      - Chat History
    security:
      - Bearer: []
    responses:
      200:
        description: Chat sessions
    """
    return jsonify({'chats': chat_store.list_chats(request.current_user['user_id'])})

@app.route('/chats/<chat_id>', methods=['GET'])
@token_required
def get_chat(chat_id):
    """
    Get a chat session and all its messages
    ---
    tags:
      - Chat History
    security:
      - Bearer: []
    parameters:
      - in: path
        name: chat_id
        type: string
        required: true
    responses:
      200:
        description: Chat with messages
      404:
        description: Chat not found
    """
    user_id = request.current_user['user_id']
    chat = chat_store.get_chat(chat_id, user_id) if valid_chat_id(chat_id) else None
    if not chat:
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
    return jsonify({'chat': chat, 'messages': chat_store.get_messages(chat_id, user_id),
                    'dashboard': dashboard_view(chat_store.get_state(chat_id, user_id))})

@app.route('/chats/<chat_id>', methods=['DELETE'])
@token_required
def delete_chat(chat_id):
    """
    Delete a chat session and its messages
    ---
    tags:
      - Chat History
    security:
      - Bearer: []
    parameters:
      - in: path
        name: chat_id
        type: string
        required: true
    responses:
      200:
        description: Deleted
      404:
        description: Chat not found
    """
    if not valid_chat_id(chat_id) or not chat_store.delete_chat(chat_id, request.current_user['user_id']):
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
    approvals_store.delete_for_chat(request.current_user['user_id'], chat_id)
    return jsonify({'success': True})

@app.route('/chats/<chat_id>/summary', methods=['POST'])
@token_required
def summarize_chat(chat_id):
    """
    Generate (and store) an AI summary of a chat session
    ---
    tags:
      - Reports
    security:
      - Bearer: []
    parameters:
      - in: path
        name: chat_id
        type: string
        required: true
    responses:
      200:
        description: Summary text
      404:
        description: Chat not found
    """
    user_id = request.current_user['user_id']
    messages = chat_store.get_messages(chat_id, user_id) if valid_chat_id(chat_id) else None
    if messages is None:
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
    if not messages:
        return jsonify({'success': True, 'summary': text("chat_summary_empty")})

    transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:1500]}" for m in messages)[-15000:]
    try:
        with usage.step("Chat summary"):
            summary = chat_llm.invoke([
                SystemMessage(content=PROMPT_SECTIONS.get("CHAT_SUMMARY_INSTRUCTIONS", "")),
                HumanMessage(content=transcript),
            ]).content.strip()
    except Exception:
        log.exception('Summary failed for %s', chat_id)
        return jsonify({'success': False, 'error': 'Could not generate summary right now'}), 502
    chat_store.update_chat(chat_id, user_id, reembed=True, summary=summary)
    return jsonify({'success': True, 'summary': summary})

@app.route('/chats/<chat_id>/agents/compare', methods=['POST'])
@token_required
def compare_agents(chat_id):
    """
    Compare the specialist agents' results with the Web Search Agent
    ---
    tags:
      - Agents
    security:
      - Bearer: []
    parameters:
      - in: path
        name: chat_id
        type: string
        required: true
    responses:
      200:
        description: Updated agent trace including the web comparison
      404:
        description: Chat or agent run not found
    """
    user_id = request.current_user['user_id']
    if not valid_chat_id(chat_id) or chat_store.get_chat_meta(chat_id, user_id) is None:
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
    state = chat_store.get_state(chat_id, user_id)
    with usage.collect():
        run = compare_agents_with_web(state)
    if not run:
        return jsonify({'success': False, 'error': text("compare_needs_plan")}), 404
    chat_store.update_chat(chat_id, user_id, state=state)
    return jsonify({'success': True, 'agent_run': run})

@app.route('/reports/search', methods=['GET'])
@token_required
def search_chats():
    """
    Semantic search over the user's chat messages (vector DB)
    ---
    tags:
      - Reports
    security:
      - Bearer: []
    parameters:
      - in: query
        name: q
        type: string
        required: true
        description: Natural-language query over your chats
      - in: query
        name: chat_id
        type: string
        required: false
        description: Limit the search to one chat
    responses:
      200:
        description: Matching messages ranked by similarity
    """
    query = (request.args.get('q') or '').strip()[:500]
    if not query:
        return jsonify({'results': []})
    scope = request.args.get('chat_id') or None
    if scope and not valid_chat_id(scope):
        return jsonify({'results': []})
    results = chat_store.search(request.current_user['user_id'], query, chat_id=scope)
    return jsonify({'results': results})

def usage_cost(totals):
    pricing = CONFIG.get("usage_pricing") or {}
    rates = pricing.get("per_million_tokens") or {}
    if not all(isinstance(rates.get(k), (int, float)) for k in ("input", "cached_input", "output")):
        return None
    fresh = max(0, totals["input"] - totals["cached"])
    amount = (fresh * rates["input"] + totals["cached"] * rates["cached_input"] + totals["output"] * rates["output"]) / 1_000_000
    return {"amount": round(amount, 4), "currency": pricing.get("currency", "")}

@app.route('/reports/usage', methods=['GET'])
@token_required
def usage_report():
    """
    LLM token usage across the user's chats (per day, per step and per chat)
    ---
    tags:
      - Reports
    security:
      - Bearer: []
    responses:
      200:
        description: Totals, last 14 days, steps and top chats, plus a cost estimate when prices are configured
    """
    report = chat_store.usage_report(request.current_user['user_id'])
    report['cost'] = usage_cost(report['totals'])
    return jsonify(report)

@app.route('/reports/stats', methods=['GET'])
@token_required
def chat_stats():
    """
    Usage statistics for the dashboard
    ---
    tags:
      - Reports
    security:
      - Bearer: []
    responses:
      200:
        description: Totals and top routes
    """
    return jsonify(chat_store.stats(request.current_user['user_id']))

@app.route('/knowledge/docs', methods=['GET'])
@token_required
def knowledge_docs():
    """
    List documents in the user's RAG knowledge base
    ---
    tags:
      - Knowledge Base (RAG)
    security:
      - Bearer: []
    responses:
      200:
        description: Uploaded documents with chunk counts
    """
    return jsonify({'documents': knowledge_store.list_documents(request.current_user['user_id'])})

@app.route('/knowledge/upload', methods=['POST'])
@token_required
def knowledge_upload():
    """
    Upload files to the RAG knowledge base (any type; text is extracted, chunked and embedded)
    ---
    tags:
      - Knowledge Base (RAG)
    security:
      - Bearer: []
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: files
        type: file
        required: true
        description: One or more files (PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, TXT, MD, code, ...)
    responses:
      200:
        description: Per-file result (chunks stored, or why a file was skipped)
    """
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files received'}), 400
    results = []
    for f in files:
        name = os.path.basename(f.filename or 'untitled')[:200]
        try:
            doc = knowledge_store.add_document(request.current_user['user_id'], name, f.read())
            results.append({'filename': name, 'success': True, **doc})
        except ValueError as e:
            results.append({'filename': name, 'success': False, 'error': str(e)})
        except Exception:
            log.exception('Knowledge upload failed for %s', name)
            results.append({'filename': name, 'success': False,
                            'error': 'could not process this file - it may be damaged or password-protected'})
    return jsonify({'success': any(r['success'] for r in results), 'results': results})

@app.route('/knowledge/docs/<doc_id>', methods=['DELETE'])
@token_required
def knowledge_delete(doc_id):
    """
    Remove a document (and all its chunks) from the knowledge base
    ---
    tags:
      - Knowledge Base (RAG)
    security:
      - Bearer: []
    parameters:
      - in: path
        name: doc_id
        type: string
        required: true
    responses:
      200:
        description: Deleted
      404:
        description: Document not found
    """
    if not valid_chat_id(doc_id) or not knowledge_store.delete_document(request.current_user['user_id'], doc_id):
        return jsonify({'success': False, 'error': 'Document not found'}), 404
    return jsonify({'success': True})

@app.errorhandler(413)
def too_large(_):
    return jsonify({'success': False, 'error': 'Upload too large - the limit is 50 MB per upload'}), 413


def plan_from_chat(chat_id, user_id, state=None):
    if not valid_chat_id(chat_id) or chat_store.get_chat_meta(chat_id, user_id) is None:
        return None, 'Chat not found'
    state = state if state is not None else chat_store.get_state(chat_id, user_id)
    record, error = adapter.trip_from_state(state)
    if record is None:
        return None, error
    messages = chat_store.get_messages(chat_id, user_id) or []
    plan_text = next((m['content'] for m in reversed(messages) if m['type'] == TYPES['plan']), '')
    return record, plan_text

def pdf_response(plan):
    return Response(adapter.build_pdf(plan), mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename="{adapter.pdf_filename(plan)}"'})

@app.route('/plans', methods=['GET'])
@token_required
def list_plans():
    """
    List the user's saved plans (their calendar)
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    responses:
      200:
        description: Plans sorted by date
    """
    return jsonify({'plans': plans_store.list_plans(request.current_user['user_id'])})

def approval_view(item, chat_titles=None):
    view = {k: item[k] for k in ("id", "chat_id", "trigger", "checkpoint", "title", "question", "options", "preview",
                                 "context", "status", "decision", "decision_label", "decided_via", "outcome",
                                 "created_at", "decided_at")}
    view["age"] = hitl.age_text(item["age_seconds"])
    view["stale_note"] = hitl.stale_note(item["age_seconds"]) if item["status"] == "pending" else ""
    view["chat_title"] = (chat_titles or {}).get(item["chat_id"], "")
    return view

@app.route('/approvals', methods=['GET'])
@token_required
def list_approvals():
    """
    Human-in-the-loop queue - steps waiting for the user's go-ahead, and past decisions
    ---
    tags:
      - Human in the loop
    security:
      - Bearer: []
    parameters:
      - in: query
        name: status
        type: string
        enum: ["pending", "completed", "all"]
        default: pending
      - in: query
        name: chat_id
        type: string
        required: false
    responses:
      200:
        description: Approvals plus the page name and note from prompt_md HUMAN_IN_THE_LOOP
    """
    user_id = request.current_user['user_id']
    status = request.args.get('status', 'pending')
    items = approvals_store.list_for_user(user_id, status=None if status == 'all' else status,
                                          chat_id=request.args.get('chat_id') or None)
    titles = {}
    for item in items:
        if item['chat_id'] not in titles:
            meta = chat_store.get_chat_meta(item['chat_id'], user_id) if valid_chat_id(item['chat_id']) else None
            titles[item['chat_id']] = (meta or {}).get('title', '')
    settings = hitl.settings()
    return jsonify({'name': settings.get('inbox_name'), 'tagline': settings.get('inbox_tagline'),
                    'pending_count': approvals_store.count_pending(user_id),
                    'approvals': [approval_view(i, titles) for i in items]})

@app.route('/approvals/<approval_id>/decide', methods=['POST'])
@token_required
def decide_approval(approval_id):
    """
    Approve or decline a pending step - the paused flow continues and its result is posted to the chat
    ---
    tags:
      - Human in the loop
    security:
      - Bearer: []
    parameters:
      - in: path
        name: approval_id
        type: string
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            action:
              type: string
              example: approve
              description: One of the option actions of this approval (see GET /approvals)
    responses:
      200:
        description: Decision applied - returns the chat reply and the updated approval
      400:
        description: Not pending any more, or unknown action
      404:
        description: Not found
    """
    from core.orchestrator import resume_approval
    user_id = request.current_user['user_id']
    username = request.current_user['username']
    item = approvals_store.get(approval_id, user_id) if re.fullmatch(r'[0-9a-f]{32}', approval_id or '') else None
    if not item:
        return jsonify({'success': False, 'error': 'Approval not found'}), 404
    if item['status'] != 'pending':
        return jsonify({'success': False, 'error': f"Already decided: {item.get('decision_label') or item['status']}."}), 400
    action = ((request.get_json(silent=True) or {}).get('action') or '').strip().lower()
    option = next((o for o in item['options'] if o['action'] == action), None)
    if not option:
        return jsonify({'success': False, 'error': 'Choose one of the offered options.'}), 400
    meta = chat_store.get_chat_meta(item['chat_id'], user_id)
    if meta is None:
        return jsonify({'success': False, 'error': 'The chat for this request was deleted.'}), 404
    state = chat_store.get_state(item['chat_id'], user_id)
    with usage.collect() as collector:
        result = resume_approval(item, action, state, 'clearances')
        if result is not None:
            result['usage'] = collector.summary()
    if result is None:
        return jsonify({'success': False, 'error': 'This request can no longer be decided.'}), 400
    data = persist_turn(item['chat_id'], user_id, username, meta, state, f"✅ {option['label']}", result)
    return jsonify({'success': True, **data, 'decided': approval_view(approvals_store.get(approval_id, user_id))})

@app.route('/runs/<run_id>/events', methods=['GET'])
@token_required
def run_events(run_id):
    """
    Live progress of one chat request as Server-Sent Events (agents, tools, tokens, memory)
    ---
    tags:
      - Chat
    security:
      - Bearer: []
    produces:
      - text/event-stream
    parameters:
      - in: path
        name: run_id
        type: string
        required: true
        description: The run_id the client sent with POST /chat (32 hex characters)
    responses:
      200:
        description: A stream of JSON events, ending with an "end" event
    """
    if not RUN_ID_RE.fullmatch(run_id or ''):
        return jsonify({'success': False, 'error': 'Invalid run id'}), 400
    stream = events.stream(run_id, request.current_user['user_id'])
    return Response(stream_with_context(stream), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/memory/items', methods=['GET'])
@token_required
def memory_items():
    """
    Long-term memory - preferences remembered about the current user
    ---
    tags:
      - Memory
    security:
      - Bearer: []
    responses:
      200:
        description: Remembered items and the categories this domain can remember
    """
    return jsonify({'enabled': memory.enabled(), 'categories': memory.CATEGORIES,
                    'items': memory.items(request.current_user['user_id'])})

@app.route('/memory/items', methods=['POST'])
@token_required
def memory_add():
    """
    Remember (or update) one preference
    ---
    tags:
      - Memory
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            key:
              type: string
              example: seat
            value:
              type: string
              example: window seat
    responses:
      200:
        description: Saved
      400:
        description: Unknown category or empty value
    """
    data = request.get_json(silent=True) or {}
    key, value = (data.get('key') or '').strip(), (data.get('value') or '').strip()
    if key not in memory.CATEGORIES or not value:
        return jsonify({'success': False, 'error': 'Choose a category and enter a value.'}), 400
    memory_store.save(request.current_user['user_id'], key, value, 'Added on the Memory page')
    return jsonify({'success': True, 'items': memory.items(request.current_user['user_id'])})

@app.route('/memory/items/<key>', methods=['DELETE'])
@token_required
def memory_forget(key):
    """
    Forget one remembered preference
    ---
    tags:
      - Memory
    security:
      - Bearer: []
    parameters:
      - in: path
        name: key
        type: string
        required: true
    responses:
      200:
        description: Forgotten
    """
    memory_store.forget(request.current_user['user_id'], key)
    return jsonify({'success': True, 'items': memory.items(request.current_user['user_id'])})

@app.route('/memory/items', methods=['DELETE'])
@token_required
def memory_forget_all():
    """
    Forget everything remembered about the current user
    ---
    tags:
      - Memory
    security:
      - Bearer: []
    responses:
      200:
        description: Number of items forgotten
    """
    count = memory_store.forget_all(request.current_user['user_id'])
    return jsonify({'success': True, 'forgotten': count, 'items': []})

def mcp_session(restart=False):
    return mcp_client.session(request.current_user['user_id'], get_token_from_header(),
                              request.host_url.rstrip('/'), restart=restart)

def mcp_reply(client, since, **data):
    return jsonify({'success': True, **data, 'log': client.log[since:]})

@app.route('/mcp/connect', methods=['POST'])
@token_required
def mcp_connect():
    """
    MCP console - start the MCP server over stdio, perform the handshake and list its tools, resources and prompts
    ---
    tags:
      - MCP
    security:
      - Bearer: []
    responses:
      200:
        description: Server info, capabilities, tools, resources, prompts and the JSON-RPC messages exchanged
      502:
        description: The MCP server could not be started
    """
    try:
        client = mcp_session(restart=bool((request.get_json(silent=True) or {}).get('restart')))
        since = 0
        tools = client.request('tools/list').get('tools', [])
        resources = client.request('resources/list').get('resources', [])
        prompts = client.request('prompts/list').get('prompts', [])
    except mcp_client.McpError as e:
        return jsonify({'success': False, 'error': str(e)}), 502
    return mcp_reply(client, since, server=client.server, tools=tools, resources=resources, prompts=prompts)

@app.route('/mcp/call', methods=['POST'])
@token_required
def mcp_call():
    """
    MCP console - call one MCP tool (tools/call) and return its result with the protocol messages
    ---
    tags:
      - MCP
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            tool:
              type: string
              example: get_memory
            arguments:
              type: object
    responses:
      200:
        description: Tool result and JSON-RPC messages
    """
    data = request.get_json(silent=True) or {}
    try:
        client = mcp_session()
        since = len(client.log)
        result = client.request('tools/call', {'name': data.get('tool', ''), 'arguments': data.get('arguments') or {}})
    except mcp_client.McpError as e:
        return jsonify({'success': False, 'error': str(e)}), 502
    return mcp_reply(client, since, result=result)

@app.route('/mcp/resource', methods=['POST'])
@token_required
def mcp_resource():
    """
    MCP console - read one MCP resource (resources/read)
    ---
    tags:
      - MCP
    security:
      - Bearer: []
    responses:
      200:
        description: Resource contents and JSON-RPC messages
    """
    data = request.get_json(silent=True) or {}
    try:
        client = mcp_session()
        since = len(client.log)
        result = client.request('resources/read', {'uri': data.get('uri', '')})
    except mcp_client.McpError as e:
        return jsonify({'success': False, 'error': str(e)}), 502
    return mcp_reply(client, since, result=result)

@app.route('/mcp/prompt', methods=['POST'])
@token_required
def mcp_prompt():
    """
    MCP console - render one MCP prompt (prompts/get)
    ---
    tags:
      - MCP
    security:
      - Bearer: []
    responses:
      200:
        description: Prompt messages and JSON-RPC messages
    """
    data = request.get_json(silent=True) or {}
    try:
        client = mcp_session()
        since = len(client.log)
        result = client.request('prompts/get', {'name': data.get('name', ''), 'arguments': data.get('arguments') or {}})
    except mcp_client.McpError as e:
        return jsonify({'success': False, 'error': str(e)}), 502
    return mcp_reply(client, since, result=result)

@app.route('/mcp/config', methods=['GET'])
@token_required
def mcp_config():
    """
    Ready-to-use configuration for external MCP clients (MCP Inspector, VS Code, desktop AI assistants)
    ---
    tags:
      - MCP
    security:
      - Bearer: []
    responses:
      200:
        description: Command, arguments and environment for this machine
    """
    server = os.path.join(ROOT, 'core', 'mcp_server.py')
    env = {'SKYMATE_URL': request.host_url.rstrip('/'), 'SKYMATE_USERNAME': request.current_user['username'],
           'SKYMATE_PASSWORD': '<your password>'}
    return jsonify({'command': sys.executable, 'args': [server], 'env': env, 'name': CONFIG['app']['name'].lower()})

@app.route('/drafts', methods=['GET'])
@token_required
def list_drafts():
    """
    Draft bookings the user saved to finish later (shown after login)
    ---
    tags:
      - Booking (demo)
    security:
      - Bearer: []
    responses:
      200:
        description: Drafts plus the DRAFT_BOOKINGS message from prompt_md
        schema:
          type: object
          properties:
            enabled:
              type: boolean
              description: False when prompt_md has no DRAFT_BOOKINGS checkpoint
            title:
              type: string
            message:
              type: string
            drafts:
              type: array
              items:
                type: object
    """
    from core.orchestrator import drafts_view
    return jsonify(drafts_view(request.current_user['user_id']))

@app.route('/plans', methods=['POST'])
@token_required
def save_plan():
    """
    Save the plan shown in a chat to the user's calendar
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            chat_id:
              type: string
    responses:
      200:
        description: Saved plan
      400:
        description: No plan in this chat yet
    """
    user_id = request.current_user['user_id']
    chat_id = ((request.get_json() or {}).get('chat_id') or '').strip()
    record, plan_text = plan_from_chat(chat_id, user_id)
    if record is None:
        return jsonify({'success': False, 'error': plan_text}), 404 if plan_text == 'Chat not found' else 400
    plan, updated = plans_store.save_plan(user_id, chat_id, record, plan_text)
    return jsonify({'success': True, 'plan': plan, 'updated': updated})

@app.route('/plans/<plan_id>', methods=['GET'])
@token_required
def get_plan(plan_id):
    """
    Get one saved plan including its full text
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    parameters:
      - in: path
        name: plan_id
        type: string
        required: true
    responses:
      200:
        description: Plan
      404:
        description: Not found
    """
    plan = plans_store.get_plan(request.current_user['user_id'], plan_id) if valid_chat_id(plan_id) else None
    if not plan:
        return jsonify({'success': False, 'error': 'Plan not found'}), 404
    return jsonify({'success': True, 'plan': plan})

@app.route('/plans/<plan_id>', methods=['DELETE'])
@token_required
def delete_plan(plan_id):
    """
    Remove a plan from the calendar
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    parameters:
      - in: path
        name: plan_id
        type: string
        required: true
    responses:
      200:
        description: Deleted
      404:
        description: Not found
    """
    if not valid_chat_id(plan_id) or not plans_store.delete_plan(request.current_user['user_id'], plan_id):
        return jsonify({'success': False, 'error': 'Plan not found'}), 404
    return jsonify({'success': True})

@app.route('/plans/<plan_id>/pdf', methods=['GET'])
@token_required
def plan_pdf_download(plan_id):
    """
    Download a saved plan as PDF
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    produces:
      - application/pdf
    parameters:
      - in: path
        name: plan_id
        type: string
        required: true
    responses:
      200:
        description: PDF file
      404:
        description: Not found
    """
    plan = plans_store.get_plan(request.current_user['user_id'], plan_id) if valid_chat_id(plan_id) else None
    if not plan:
        return jsonify({'success': False, 'error': 'Plan not found'}), 404
    return pdf_response(plan)

@app.route('/chats/<chat_id>/itinerary.pdf', methods=['GET'])
@token_required
def chat_plan_pdf(chat_id):
    """
    Download the plan currently shown in a chat as PDF
    ---
    tags:
      - Calendar
    security:
      - Bearer: []
    produces:
      - application/pdf
    parameters:
      - in: path
        name: chat_id
        type: string
        required: true
    responses:
      200:
        description: PDF file
      400:
        description: No plan yet
    """
    user_id = request.current_user['user_id']
    record, plan_text = plan_from_chat(chat_id, user_id)
    if record is None:
        return jsonify({'success': False, 'error': plan_text}), 404 if plan_text == 'Chat not found' else 400
    plan_id = plans_store.find_for_chat(user_id, chat_id, record[ID])
    saved = plans_store.get_plan(user_id, plan_id) if plan_id else None
    plan = {**record, **(saved or {}), 'itinerary': plan_text or (saved or {}).get('itinerary', '')}
    return pdf_response(plan)


@app.route('/bookings', methods=['POST'])
@token_required
def create_booking():
    """
    Demo payment gateway - "pay" for a plan and mark it booked (no real payment is taken)
    ---
    tags:
      - Booking (demo)
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            plan_id:
              type: string
            chat_id:
              type: string
              description: Used when the plan isn't saved yet - it is saved first
            travellers:
              type: array
              items:
                type: string
            email:
              type: string
            card_number:
              type: string
              example: "4111 1111 1111 1111"
            card_expiry:
              type: string
              example: "12/28"
            card_cvv:
              type: string
              example: "123"
    responses:
      200:
        description: Booked - returns the updated plan with a booking reference
      400:
        description: Validation error or declined (demo)
    """
    user_id = request.current_user['user_id']
    data = request.get_json() or {}
    plan_id = (data.get('plan_id') or '').strip()
    if not plan_id and data.get('chat_id'):
        record, plan_text = plan_from_chat(data['chat_id'].strip(), user_id)
        if record is None:
            return jsonify({'success': False, 'error': plan_text}), 400
        plan_id = plans_store.save_plan(user_id, data['chat_id'].strip(), record, plan_text)[0]['plan_id']
    plan = plans_store.get_plan(user_id, plan_id, with_text=False) if valid_chat_id(plan_id) else None
    if not plan:
        return jsonify({'success': False, 'error': 'Plan not found'}), 404
    if plan.get('status') == 'booked':
        return jsonify({'success': False, 'error': f"Already booked (reference {plan['booking_ref']})."}), 400

    error, booking = adapter.validate_commit(plan, data)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    booked = plans_store.mark_booked(user_id, plan_id, booking)
    return jsonify({'success': True, 'plan': booked})

@app.route('/me')
@token_required
def me():
    """
    Get Current User Info
    ---
    tags:
      - Authentication
    security:
      - Bearer: []
    responses:
      200:
        description: Current user information
        schema:
          type: object
          properties:
            user_id:
              type: string
            username:
              type: string
      401:
        description: Unauthorized
    """
    user = get_current_user()
    return jsonify({
        'user_id': user['user_id'],
        'username': user['username']
    })

if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 5000
    BASE_URL = f'http://{HOST}:{PORT}'

    def print_startup_info():
        print("\n" + "=" * 70)
        print(f"  {CONFIG['app']['server_banner']}")
        print("=" * 70)
        print(f"  Login Page:      {BASE_URL}/login")
        print(f"  Chat Page:       {BASE_URL}/chat-page")
        print(f"  Dashboard:       {BASE_URL}/dashboard")
        print(f"  Reports:         {BASE_URL}/reports")
        print(f"  Knowledge (RAG): {BASE_URL}/knowledge")
        print(f"  Calendar:        {BASE_URL}/calendar")
        print(f"  Swagger UI:      {BASE_URL}/apidocs/")
        print(f"  Health Check:    {BASE_URL}/login")
        print("-" * 70)
        print("  API ENDPOINTS:")
        print(f"  POST   {BASE_URL}/login        - Login / Signup")
        print(f"  POST   {BASE_URL}/refresh      - Refresh access token")
        print(f"  POST   {BASE_URL}/logout       - Logout (revoke tokens)")
        print(f"  POST   {BASE_URL}/chat         - Chat with AI (requires auth)")
        print(f"  GET    {BASE_URL}/chats        - List chat sessions")
        print(f"  GET    {BASE_URL}/chats/<id>   - Chat transcript")
        print(f"  GET    {BASE_URL}/reports/search?q=  - Semantic search over chats")
        print(f"  GET    {BASE_URL}/me           - Current user info (requires auth)")
        print("-" * 70)
        print("  Swagger Documentation: {BASE_URL}/apidocs/")
        print("=" * 70)
        print("  Press CTRL+C to stop server")
        print("=" * 70 + "\n")

    def shutdown_server():
        print("\n" + "=" * 70)
        print("  SERVER SHUTTING DOWN...")
        print("=" * 70 + "\n")

    atexit.register(shutdown_server)

    def signal_handler(sig, frame):
        shutdown_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_startup_info()
    app.run(debug=True, host=HOST, port=PORT)