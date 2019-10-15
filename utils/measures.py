
def alignment_diagonal_score(alignments):
    """
    Compute how diagonal alignment predictions are. It is useful
    to measure the alignment consistency of a model
    Args:
        alignments (numpy.Array): batch of alignments.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    return alignments.max(axis=0).mean(axis=1).mean(axis=0)
