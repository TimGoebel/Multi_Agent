import streamlit as st
from openai import OpenAI

def evaluate_fine_tuned_model(api_key, fine_tuned_model_id, base_model="gpt-3.5-turbo"):
    """
    Evaluates the fine-tuned model by running test samples, comparing responses with expected outputs,
    and computing accuracy.
    
    Parameters:
    - api_key (str): OpenAI API key
    - fine_tuned_model_id (str): ID of the fine-tuned model
    - base_model (str): (Optional) Base model for comparison (default: "gpt-3.5-turbo")
    """
    st.title("Fine-Tuned Model Evaluation")

    client = OpenAI(api_key=api_key)

    # Example test samples
    test_samples = [
        {"prompt": "What is the capital of France?", "expected_output": "Paris"},
        {"prompt": "Explain Newton’s first law of motion.", "expected_output": "An object remains at rest or in motion unless acted upon by an external force."},
        {"prompt": "What is the largest planet in our solar system?", "expected_output": "Jupiter"},
        {"prompt": "Who wrote the play Romeo and Juliet?", "expected_output": "William Shakespeare"},
        {"prompt": "What is the boiling point of water in Celsius?", "expected_output": "100"},
    ]

    results = []

    st.subheader("Running Evaluation...")

    for sample in test_samples:
        try:
            # Query Fine-Tuned Model (Use chat completion endpoint)
            fine_tuned_response = client.chat.completions.create(
                model=fine_tuned_model_id,
                messages=[{"role": "user", "content": sample["prompt"]}],
                max_tokens=100,
                temperature=0
            )
            fine_tuned_output = fine_tuned_response.choices[0].message.content.strip()

            # Query Base Model for Comparison (Use chat completion endpoint)
            base_response = client.chat.completions.create(
                model=base_model,
                messages=[{"role": "user", "content": sample["prompt"]}],
                max_tokens=100,
                temperature=0
            )
            base_output = base_response.choices[0].message.content.strip()

            results.append({
                "Prompt": sample["prompt"],
                "Expected": sample["expected_output"],
                "Fine-Tuned Output": fine_tuned_output,
                "Base Model Output": base_output
            })

        except Exception as e:
            st.error(f"Error processing evaluation: {str(e)}")
            return

    # Display Evaluation Results
    st.subheader("Evaluation Results")
    correct = 0

    for r in results:
        is_correct = r["Fine-Tuned Output"].lower() == r["Expected"].lower()
        if is_correct:
            correct += 1

        st.write(f"**Prompt:** {r['Prompt']}")
        st.write(f"✅ **Expected Output:** {r['Expected']}")
        st.write(f"🤖 **Fine-Tuned Model Output:** {r['Fine-Tuned Output']}")
        st.write(f"🧠 **Base Model Output:** {r['Base Model Output']}")
        st.write(f"🔎 **Correct?** {'✅ Yes' if is_correct else '❌ No'}")
        st.write("---")

    # Compute Model Accuracy
    accuracy = (correct / len(results)) * 100
    st.success(f"🎯 Fine-Tuned Model Accuracy: {accuracy:.2f}%")

    # Model Summary
    st.subheader("Model Performance Summary")
    st.write(f"**Fine-Tuned Model ID:** {fine_tuned_model_id}")
    st.write(f"**Base Model Used for Comparison:** {base_model}")
    st.write(f"**Total Test Cases:** {len(test_samples)}")
    st.write(f"**Correct Predictions:** {correct}/{len(test_samples)}")

    return accuracy
