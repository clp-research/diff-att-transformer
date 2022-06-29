import math

import torch
from torch.nn import functional as F
from tqdm import tqdm


class TestContext:

    def __init__(self, output_file_path, word_map, rev_word_map, beam_size, nb_heads):
        self.output_file_path = output_file_path
        self.nb_heads = nb_heads
        self.beam_size = beam_size
        self.rev_word_map = rev_word_map
        self.word_map = word_map


@torch.no_grad()
def translate(context: TestContext, model, image_feature_encoder, device, img_before, img_after,
              length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.
    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    special_tokens = {context.word_map['<start>'],
                      context.word_map['<end>'],
                      context.word_map['<pad>']}

    # Beam size
    k = context.beam_size

    # Minimum number of hypotheses to complete
    n_completed_hypotheses = min(k, 10)

    # Vocab size
    vocab_size = len(context.word_map)

    # Move to GPU, if available
    img_before = img_before.to(device)
    img_after = img_after.to(device)

    # Extract Image Features/Image Encoder
    img_before = image_feature_encoder(img_before)  # == source_sequences
    img_after = image_feature_encoder(img_after)  # == source_sequences
    image_embedding_attention_heads = context.nb_heads
    source_sequence_lengths = torch.LongTensor([image_embedding_attention_heads]).to(
        device)  # (1) source sequence length specifies the number of created visual words = nb_heads
    source_sequence_lengths = source_sequence_lengths.to(device)  # (N) => in this case it's just one so N=1

    # Encode
    encoder_sequences, img_before_alphas, img_after_alphas = model.encoder(img_before, img_after,
                                                                           encoder_sequence_lengths=source_sequence_lengths)  # (1, source_sequence_length, d_model)

    # Our hypothesis to begin with is just <start>
    hypotheses = torch.LongTensor([[context.word_map['<start>']]]).to(device)  # (1, 1)
    hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # (1)

    # Tensor to store hypotheses' scores; now it's just 0
    hypotheses_scores = torch.zeros(1).to(device)  # (1)

    # Lists to store completed hypotheses and their scores
    completed_hypotheses = list()
    completed_hypotheses_scores = list()

    # Start decoding
    step = 1

    # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
    # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
    while True:
        s = hypotheses.size(0)
        decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                          decoder_sequence_lengths=hypotheses_lengths,
                                          encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                          encoder_sequence_lengths=source_sequence_lengths.repeat(
                                              s))  # (s, step, vocab_size)

        # Scores at this step
        scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

        # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
        scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

        # Unroll and find top k scores, and their unrolled indices
        top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

        # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
        prev_word_indices = unrolled_indices // vocab_size  # (k)
        next_word_indices = unrolled_indices % vocab_size  # (k)

        # Construct the the new top k hypotheses from these indices
        top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                     dim=1)  # (k, step + 1)

        # Which of these new hypotheses are complete (reached <EOS>)?
        complete = next_word_indices == context.word_map['<end>']  # (k), bool

        # Set aside completed hypotheses and their scores normalized by their lengths
        # For the length normalization formula, see
        # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
        completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
        norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
        completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

        # Stop if we have completed enough hypotheses
        if len(completed_hypotheses) >= n_completed_hypotheses:
            break

        # Else, continue with incomplete hypotheses
        hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
        hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
        hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

        # Stop if things have been going on for too long
        if step > 50:
            break
        step += 1

    # If there is not a single completed hypothesis, use partial hypotheses
    if len(completed_hypotheses) == 0:
        completed_hypotheses = hypotheses.tolist()
        completed_hypotheses_scores = hypotheses_scores.tolist()

    # Get attention weights of the encoder-decoder-multi-head-attention layer
    # get the enc-dec attention weights of the last decoder layer
    encoder_decoder_mh_att_weights = model.decoder.decoder_layers[0][1].att_weights

    # Decode the hypotheses
    all_hypotheses = list()
    for i, h in enumerate(list(
            map(lambda c: [context.rev_word_map[w] for w in c if w not in special_tokens],
                completed_hypotheses))):
        all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

    # Find the best scoring completed hypothesis
    i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
    best_hypothesis = all_hypotheses[i]["hypothesis"]

    return best_hypothesis, all_hypotheses, img_before_alphas, encoder_decoder_mh_att_weights  # img_after_alphas


@torch.no_grad()
def evaluate(context: TestContext, data_loader, model, image_feature_encoder, device):
    # DataLoader

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    special_tokens = {context.word_map['<start>'],
                      context.word_map['<end>'],
                      context.word_map['<pad>']}
    task_desc = "EVALUATING AT BEAM SIZE " + str(context.beam_size)
    for img_before, img_after, img_references in tqdm(data_loader, desc=task_desc):
        img_references = img_references[0]  # there is only a single sample in the batch
        img_references = img_references.tolist()  # convert long tensor to list
        img_references = list(
            map(lambda c: [context.rev_word_map[w] for w in c if w not in special_tokens], img_references))
        references.append(img_references)

        best_hypothesis, _, _, _ = translate(context, model, image_feature_encoder, device,
                                             img_before, img_after,
                                             length_norm_coefficient=0.6)

        # Hypotheses
        hypotheses.append([w for w in best_hypothesis if w not in special_tokens])
        assert len(references) == len(hypotheses)

    return references, hypotheses
