import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model():
    """
    Loads the fine-tuned model from Hugging Face Hub and runs inference.
    """
    model_hub_id = "jasong03/qwen3-4b-emo-plain-sft"

    print(f"Loading model: {model_hub_id}")

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_hub_id)
    tokenizer = AutoTokenizer.from_pretrained(model_hub_id)

    # Set the model to evaluation mode
    model.eval()

    # Sample text to classify
    sample_text = "I'm so happy and excited to see this project succeed!"
    print(f"\nInput text: {sample_text}")

    # Format the input
    prompt = f"Input: {sample_text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nGenerating response...")

    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=20)
    
    # Print the result
    print("\nModel output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_model()
