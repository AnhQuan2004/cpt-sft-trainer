import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def get_true_labels(example):
    """Extracts true emotion labels from a dataset example."""
    emotions = []
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    for col in emotion_columns:
        if col in example and example[col] == 1:
            emotions.append(col)
    return sorted(emotions) if emotions else ["neutral"]

def evaluate_model():
    """
    Loads the fine-tuned model and evaluates it on the GoEmotions test set.
    """
    model_hub_id = "jasong03/qwen3-4b-emo-plain-sft"
    dataset_name = "SetFit/go_emotions"

    print(f"Loading model: {model_hub_id}")
    model = AutoModelForCausalLM.from_pretrained(model_hub_id)
    tokenizer = AutoTokenizer.from_pretrained(model_hub_id)
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")

    print(f"Loading test split of dataset: {dataset_name}")
    test_dataset = load_dataset(dataset_name, split="test")

    correct_predictions = 0
    total_predictions = len(test_dataset)

    print("Running evaluation...")
    for i in tqdm(range(total_predictions)):
        example = test_dataset[i]
        text = example["text"]
        
        # Prepare the prompt for the model
        prompt = f"Input: {text}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate the model's prediction
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prediction part of the output
        prediction_text = decoded_output.split("Output:")[1].strip()
        predicted_labels = sorted([label.strip() for label in prediction_text.split(',')])

        # Get the ground truth labels
        true_labels = get_true_labels(example)

        # Compare prediction with true labels
        if predicted_labels == true_labels:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nEvaluation Complete.")
    print(f"Total examples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
