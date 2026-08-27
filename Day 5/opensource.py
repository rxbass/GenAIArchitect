from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Local model path
MODEL_PATH = "./models/qwen-0.5b"  # Change if needed

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cpu"
)

print("Model loaded successfully!")

while True:
    user_input = input("\nYou: Hello dude! How are you? (Type 'exit' or 'quit' to stop)\n> ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\nAssistant:", response)