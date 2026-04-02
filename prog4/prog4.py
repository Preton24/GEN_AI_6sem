# program_4.py

import gensim.downloader as api
import google.generativeai as genai


# 🔑 Directly paste your API key here
GOOGLE_API_KEY = "AIzaSyAxa5OVBMqtcPdkA8xZ8iNSoc8egGll7Ng"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)


# Load Word2Vec model (downloads on first run)
word2vec_model = api.load("word2vec-google-news-300")


def get_similar_words(word, top_n=3):
    try:
        return [w for w, _ in word2vec_model.most_similar(word, topn=top_n)]
    except KeyError:
        return [word]


def main():
    original_prompt = "Generate a detailed story about an astronaut exploring a distant exoplanet."
    keywords = ["astronaut", "exploring", "distant", "exoplanet"]

    expanded_prompt = original_prompt

    for word in keywords:
        similar_words = get_similar_words(word)
        expanded_prompt = expanded_prompt.replace(
            word, f"{word} ({','.join(similar_words)})"
        )

    print("Original Prompt:", original_prompt)
    print("Enriched Prompt:", expanded_prompt)

    # Initialize Gemini model
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate responses
    response_original = gemini_model.generate_content(original_prompt)
    response_enriched = gemini_model.generate_content(expanded_prompt)

    print("\nResponse to Original Prompt:\n", response_original.text)
    print("\nResponse to Enriched Prompt:\n", response_enriched.text)


if __name__ == "__main__":
    main()