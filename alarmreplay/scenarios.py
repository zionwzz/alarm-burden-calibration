"""The simulation scenarios as the manuscript letters them.

The simulation code keys its scenarios A, B, C, D, E, F, G, I, J (H is the patient-count sweep),
and the seeds, the saved results and the replicate files are all keyed by those letters, so they
cannot change. The manuscript letters the same scenarios A to I in the order of its design table,
which groups them by the condition each examines. This module holds that correspondence once, and
the tables and figures print the manuscript's letters through it.
"""

#: simulation key -> manuscript letter, in the order of the design table
PAPER_LETTER = {"A": "A", "G": "B", "D": "C", "B": "D", "C": "E", "I": "F", "J": "G", "E": "H", "F": "I"}

#: the manuscript's order of the simulation keys
PAPER_ORDER = ["A", "G", "D", "B", "C", "I", "J", "E", "F"]

#: short names as the manuscript prints them (the sparse scenario is named after the one it copies)
SHORT_NAME = {"A": "Transport holds", "G": "Stronger patient dependence", "D": "Pooled-cell heterogeneity",
              "B": "Overshoot and composition shift", "C": "Sparse feature cells",
              "I": "Duration-dependent zero inflation", "J": "Feature-dependent zero inflation",
              "E": "Incomplete support", "F": "Reward transport fails"}


def paper(key):
    """The manuscript letter of a simulation key."""
    return PAPER_LETTER[key]


def letters(keys):
    """Manuscript letters of several simulation keys, in manuscript order, as 'A, B and C'."""
    ls = sorted(PAPER_LETTER[k] for k in keys)
    return ls[0] if len(ls) == 1 else ", ".join(ls[:-1]) + " and " + ls[-1]
