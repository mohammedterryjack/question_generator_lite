from os import environ

from flask import Flask,request
from werkzeug.exceptions import BadRequest

from src.semantic_captioning_character_lstm_decoder import SemanticCaptioningCharacterLSTMDecoder
   
app = Flask(__name__)
app.decoder = SemanticCaptioningCharacterLSTMDecoder(
    recursion_depth=100,
    path_to_model_weights="pretrained_models/intent_captioning_daily_dialog.hdf5"
)

@app.route("/")
def home():
    return "Universal Sentence Decoder"

@app.route(f"/generate",methods=["POST"])
def parse():
    
    if request.is_json:
        input_json = request.get_json()
        text = input_json.get("text", "")
        top_percentage = input_json.get("top_p")
        temperature = input_json.get("temperature")
        return {
            "text":text,
            "generated_reply":app.decoder.generate(text,temperature=temperature,top_p=top_percentage)
        }
    raise BadRequest()
    

if __name__ == '__main__':
    app.run(threaded=True, port=environ.get('PORT'), debug=True)