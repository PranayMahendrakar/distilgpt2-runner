from transformers import pipeline

def main():
    print("Loading DistilGPT2 model...")
    generator = pipeline("text-generation", model="distilgpt2")

    prompt = "Who are you?"

    print(f"Prompt: {prompt}")
    print("Generating text...")

    results = generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

    generated_text = results[0]["generated_text"]

    print("\n=== Generated Text ===")
    print(generated_text)
    print("======================")

if __name__ == "__main__":
    main()
