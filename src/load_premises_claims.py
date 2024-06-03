"""Load the premises and claims from the Excel file with the evaluation of the texts and classify them as premises, claims or not arguments."""

import csv
from load_arguments import take_n_random_elements


class Corpus_premises_claims:
    """Class to store the premises and claims and their labels. The labels are 2 if the text is a premise, 1 if it is a claim of an argument and 0 if it is not an argument."""

    def __init__(self):
        self.data = []

    def append(self, proposal_id, argument_id, premise_claim, label):
        if premise_claim[0] == " ":
            premise_claim = premise_claim[1:]
        self.data.append((proposal_id, argument_id, premise_claim, label))

    def save_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            for proposal_id, argument_id, premise_claim, label in self.data:
                file.write(
                    str(proposal_id)
                    + "\t"
                    + str(argument_id)
                    + "\t"
                    + str(premise_claim)
                    + "\t"
                    + str(label)
                    + "\n"
                )


def load_premises_claims(file_path, output_file_path):
    """Classify the texts as premises, claims or not arguments based on the value of the field 'Claim+premise?WHY?' of the evaluation of the texts."""
    corpus = Corpus_premises_claims()
    not_arguments = []
    n_arguments = 0
    with open(file_path, "r", newline="", encoding="latin-1") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            if row["Tipo elemento"] != "Argumento":
                continue
            proposal_id = row["Id propuesta"]
            argument_id = row["Id elemento"]
            if row["Claim+premise?\nWHY?"] != "1" or row["Claim?"] != "1":
                not_arguments.append((proposal_id, argument_id, row["Valor"]))
            else:

                premise = row["Premisa"]
                claim = row["Claim"]
                corpus.append(proposal_id, argument_id, premise, 2)
                corpus.append(proposal_id, argument_id, claim, 1)
                n_arguments += 1
    not_arguments = take_n_random_elements(not_arguments, n_arguments)
    for proposal_id, argument_id, not_argument in not_arguments:
        corpus.append(proposal_id, argument_id, not_argument, 0)
    corpus.save_to_file(output_file_path)
    return corpus


if __name__ == "__main__":
    load_premises_claims("ChatGPT-evaluation.txt", "arguments_premises_claims.txt")
