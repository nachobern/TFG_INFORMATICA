from load_arguments import create_complete_corpus


def main():
    """Create a corpus with all the arguments and non-arguments. Save the corpus in a file with the specified fields."""
    corpus = create_complete_corpus()
    corpus_premise_validation = corpus.filter_by("premise_validation", [0, 1, 2])
    corpus_premise_validation.save_to_file(
        "arguments_premise_validation.txt", ["premise_validation"], replace_none="0"
    )
    corpus_coherence = corpus.filter_by("coherence", [0, 1, 2])
    corpus_coherence.save_to_file(
        "arguments_coherence.txt", ["coherence"], replace_none="0"
    )
    corpus_consistence = corpus.filter_by("consistence", [0, 1, 2])
    corpus_consistence.save_to_file(
        "arguments_consistence.txt", ["consistence"], replace_none="0"
    )
    corpus_persuasion = corpus.filter_by("persuasion", [0, 1, 2])
    corpus_persuasion.save_to_file(
        "arguments_persuasion.txt", ["persuasion"], replace_none="0"
    )
    corpus_emotional_ethic = corpus.filter_by("emotional_ethic", [0, 1, 2])
    corpus_emotional_ethic.save_to_file(
        "arguments_emotional_ethic.txt", ["emotional_ethic"], replace_none="0"
    )


if __name__ == "__main__":
    main()
