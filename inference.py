import transformers
from transformers import AutoTokenizer

def inference_example(model_path, message):
    '''
    Args:
        model_path (str): Path to the model on Huggingface to be used for inference.
        message (list): List of dictionaries containing the message history.
            Format: [{"role": "system" or "user", "content": "message"}]
            message = [
                {"role": "system", "content": "You are a helpful assistant chatbot."},
                {"role": "user", "content": "What is a Large Language Model?"}
            ]
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(message, 
                                           add_generation_prompt=True, 
                                           tokenize=False)

    # Create pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer
    )

    # Generate text
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=200,
    )
    print(sequences[0]['generated_text'])
