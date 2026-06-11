#!/usr/bin/env python3
"""
Simple chatbot application with Flask and Qwen3 model.
"""

import logging
import sys
import os

# Add the parent directory to Python path so we can import our core module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, send_from_directory
from core import LLMService, SamplingConfig, EngineArgs
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Initialize LLM service
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "Qwen3-0.6B")

engine_args = EngineArgs(
    model_path=model_path,
    device="cuda",
    num_blocks=400,
    block_size=16,
)

llm_service = LLMService.from_engine_args(engine_args)
logging.info(f"Model loaded successfully on device: {llm_service.device}")

# Initialize tokenizer for Jinja2 template
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define system prompt
system_prompt = "你是一个名为 Nanoserve 的人工智能助手。你擅长逻辑推理和结构化表达。\n\n在回答用户问题前，请务必先在 <think> 标签内进行深思熟虑，分析用户的真实意图、所需的知识点以及回答的逻辑架构。\n思考完成后，请在 <think> 标签外给出正式、专业且简洁的回答。\n\n你的目标是：逻辑严密，事实准确，语气友好。"

# Create sampling config
sampling_config = SamplingConfig(
    temperature=0.6,
    top_p=0.9,
    max_new_tokens=4096,
)

# Store conversation history
conversations = {}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversation_id', 'default')
    max_new_tokens = data.get('max_new_tokens', 4096)  # 从前端接收最大token数
    
    # Get existing conversation history or create new
    history = conversations.get(conversation_id, [])
    
    # Build messages list with system prompt and history
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add historical messages
    for msg in history:
        messages.append({"role": "user", "content": msg['user']})
        messages.append({"role": "assistant", "content": msg['assistant']})
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Apply chat template using Jinja2
    rendered_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate response
    try:
        # 创建临时的 sampling config，使用前端传入的 max_new_tokens
        temp_sampling_config = SamplingConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )
        
        generated = llm_service.generate(
            prompts=[rendered_prompt],
            sampling_config=temp_sampling_config,
        )[0]
        
        # Update conversation history
        history.append({
            'user': message,
            'assistant': generated
        })
        conversations[conversation_id] = history
        
        return jsonify({
            'response': generated,
            'conversation_id': conversation_id
        })
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    conversation_id = request.args.get('conversation_id', 'default')
    history = conversations.get(conversation_id, [])
    return jsonify(history)

@app.route('/api/clear', methods=['POST'])
def clear_history():
    conversation_id = request.json.get('conversation_id', 'default')
    if conversation_id in conversations:
        del conversations[conversation_id]
    return jsonify({'status': 'success'})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
