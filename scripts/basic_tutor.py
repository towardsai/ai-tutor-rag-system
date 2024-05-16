import sys
import os
from openai import OpenAI

# Retrieve your OpenAI API key from the environment variables and activate the OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def ask_ai_tutor(question):

    # Check if OpenAI key has been correctly added
    if not openai_api_key:
        return "OpenAI API key not found in environment variables."

    try:

        # Formulating the system prompt
        system_prompt = (
            "You are an AI tutor specialized in answering artificial intelligence-related questions. "
            "Only answer AI-related question, else say that you cannot answer this question."
        )

        # Combining the system prompt with the user's question
        prompt = f"Please provide an informative and accurate answer to the following question.\nQuestion: {question}\nAnswer:"

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        # Return the AI's response
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred: {e}"


def main():
    # Check if a question was provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py 'Your AI-related question'")
        sys.exit(1)

    # The user's question is the first command-line argument
    user_question = sys.argv[1]

    # Get the AI's response
    ai_response = ask_ai_tutor(user_question)

    # Print the AI's response
    print(f"AI Tutor says: {ai_response}")


if __name__ == "__main__":
    main()
