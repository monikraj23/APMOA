from src.rag.knowledge_base import (
    APMOAKnowledgeBase,
)


def main():

    print("\n" + "=" * 60)
    print("APMOA RAG TEST")
    print("=" * 60)

    knowledge_base = (
        APMOAKnowledgeBase()
    )

    knowledge_base.build()

    query = (
        "What do fitness and precision "
        "mean in process model evaluation?"
    )

    print("\nQuery:")
    print(query)

    results = knowledge_base.retrieve(
        query,
        top_k=2,
    )

    print("\nRetrieved Knowledge:")
    print("-" * 60)

    for result in results:

        print(
            f"\nSource: {result['source']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            result["text"]
        )

    print("\n" + "=" * 60)
    print("RAG TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()