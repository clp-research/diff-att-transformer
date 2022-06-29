import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.
    See "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps=0.1):
        """
        :param eps: smoothing co-efficient
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, lengths):
        """
        Forward prop.
        :param inputs: decoded target language sequences, a tensor of size (N, pad_length, vocab_size)
        :param targets: gold target language sequences, a tensor of size (N, pad_length)
        :param lengths: true lengths of these sequences, to be able to ignore pads, a tensor of size (N)
        :return: mean label-smoothed cross-entropy loss, a scalar
        """
        # overcome torch cuda tensor handling error => pack the padded sequence into a wrap, more info here: https://discuss.pytorch.org/t/error-with-lengths-in-pack-padded-sequence/35517/12
        # or here: https://github.com/pytorch/pytorch/issues/16542
        torch.set_default_tensor_type(torch.FloatTensor)
        # Remove pad-positions and flatten
        inputs, _, _, _ = pack_padded_sequence(input=inputs,
                                               lengths=lengths,
                                               batch_first=True,
                                               enforce_sorted=False)  # (sum(lengths), vocab_size)
        targets, _, _, _ = pack_padded_sequence(input=targets,
                                                lengths=lengths,
                                                batch_first=True,
                                                enforce_sorted=False)  # (sum(lengths))
        if self.device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        inputs.to(self.device)
        targets.to(self.device)

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = torch.zeros_like(inputs).scatter(dim=1,
                                                         index=targets.unsqueeze(1),
                                                         value=1.).to(self.device)  # (sum(lengths), n_classes), one-hot
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(
            1)  # (sum(lengths), n_classes), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1)  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
