import json
import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2024-08-01-preview"

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    try:
        data = request.get_json()
        text = data.get("values", [])[0]["data"]["text"]
        response = openai.Embedding.create(
            input=text,
            engine="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return jsonify({
            "values": [
                {
                    "recordId": data["values"][0]["recordId"],
                    "data": {
                        "embedding": embedding
                    }
                }
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
