from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Set the padding token ID
tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_reply', methods=['POST'])
def get_reply():
    user_input = request.form['user_input']
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True,
            min_length=10
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    app.run()