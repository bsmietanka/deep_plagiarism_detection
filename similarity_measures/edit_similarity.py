from typing import Sequence
from edit_distance import edit_distance

class EditSimilarity:

    def __call__(self, f1: Sequence, f2: Sequence) -> float:
        return 1 - (edit_distance(f1, f2)[0] / max(len(f1), len(f2)))

if __name__ == "__main__":
    ed = EditSimilarity()

    seq1 = ["CAT", "DOG", "BIRD"]
    seq2 = ["CAT", "DOG", "PARROT"]
    assert ed(seq1, seq2) == 1 / 3
