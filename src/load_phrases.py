"""Load the phrases from the evaluation of the arguments of the exported Excel file, and classify them as arguments or not arguments based on the value of the field 'Claim+premise?FraseWHY?' of the evaluation of the phrase. The phrases are saved to a file with two columns: the text and the label. The labels are 1 if the phrase is an argument and 0 if it is not."""

import csv
from load_arguments import take_n_random_elements


class Corpus_phrases:
    """Class to store the phrases and their labels. The labels are 1 if the phrase is an argument and 0 if it is not."""

    def __init__(self):
        """Initialize the data array and the id of the text."""
        self.data = []

    def append(self, proposal_id, argument_id, text, label):
        """Append a new text with its label to the data array."""
        if text[0] == " ":
            text = text[1:]
        self.data.append((proposal_id, argument_id, text, label))

    def save_to_file(self, filename):
        """Save the data to a file. The file has four columns: proposal_id, argument_id, text, label."""
        with open(filename, "w", encoding="utf-8") as file:
            for proposal_id, argument_id, text, label in self.data:
                file.write(
                    str(proposal_id)
                    + "\t"
                    + str(argument_id)
                    + "\t"
                    + str(text)
                    + "\t"
                    + str(label)
                    + "\n"
                )


def load_phrases(file_path, output_file_path):
    """Classify the phrases as arguments or not arguments based on the value of the field 'Claim+premise?FraseWHY?' of the evaluation of the phrases."""
    corpus = Corpus_phrases()
    not_arguments = []
    with open(file_path, "r", newline="", encoding="latin-1") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            if row["Tipo elemento"] != "Argumento":
                continue
            n_phrases = 1
            proposal_id = row["Id propuesta"]
            argument_id = row["Id elemento"]
            phrase = row["Frase" + str(n_phrases)]
            while phrase != "" and n_phrases <= 5:
                phrase = row["Frase" + str(n_phrases)]
                is_argument = row["Claim+premise?Frase" + str(n_phrases) + "\nWHY?"]
                if phrase == "" or phrase == " " or phrase == "  ":
                    break
                if is_argument == "1":
                    corpus.append(proposal_id, argument_id, phrase, is_argument)
                else:
                    not_arguments.append((proposal_id, argument_id, phrase))
                n_phrases += 1

    not_arguments = take_n_random_elements(not_arguments, len(corpus.data))
    for proposal_id, argument_id, not_argument in not_arguments:
        corpus.append(proposal_id, argument_id, not_argument, 0)
    corpus.save_to_file(output_file_path)


if __name__ == "__main__":
    load_phrases("ChatGPT-evaluation.txt", "arguments_phrases.txt")
