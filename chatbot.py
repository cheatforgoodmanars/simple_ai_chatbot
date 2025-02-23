from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
while True:
    # Encoding the conversation history \Create conversation history string
    history_string = "\n".join(conversation_history)

    # # Fetch prompt from user
    # input_text ="hello, how are you doing?"
    # Tokenization of user prompt and chat history

    # Get the input data from the user
    input_text = input("> ")
    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    print(inputs)

    tokenizer.pretrained_vocab_files_map

    # Generate output from the model
    outputs = model.generate(**inputs)
    print(outputs)

    # decode output
    response= tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    # Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    print(conversation_history)

