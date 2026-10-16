from src.llm.groq_provider import GroqProvider


def main():

    print("=" * 60)
    print("APMOA - GROQ PROVIDER TEST")
    print("=" * 60)

    print("\nInitializing Groq provider...")

    provider = GroqProvider()

    print(f"Model: {provider.model}")

    print("\nSending test request...")

    response = provider.generate(
        prompt=(
            "Explain process mining in exactly "
            "two sentences."
        ),
        system_prompt=(
            "You are an AI assistant helping "
            "with an autonomous process mining "
            "research project."
        ),
    )

    print("\n" + "-" * 60)
    print("GROQ RESPONSE")
    print("-" * 60)

    print(response)

    print("\n" + "=" * 60)
    print("GROQ PROVIDER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()