# IMPORTS
import torch.optim
import torch.utils.data
import time
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
import sys

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import CaptionDataset
from utils.train_utils import LabelSmoothedCE
from utils.transformer_utils import adjust_learning_rate, save_checkpoint, AverageMeter

sys.path.append('.')


def perform_training(config, models, optimizers, device, logger: SummaryWriter):
    """
    Training and validation.
    """
    # Initialize data-loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Normalization transform

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(config.data_folder, config.data_name, 'TRAIN', transform=transforms.Compose([normalize]),
                       img_return_mode='SPLITTED'),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(config.data_folder, config.data_name, 'VAL', transform=transforms.Compose([normalize]),
                       img_return_mode='SPLITTED'),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)

    # Loss function
    criterion = LabelSmoothedCE(eps=config.label_smoothing)

    # Move to default device
    model = models["transformer"].to(device)
    image_feature_encoder = models["resnet"].to(device)
    criterion = criterion.to(device)

    # logger.add_graph(model)

    # Find total epochs to train
    epochs = config.max_epochs
    print('Total epochs to train: ' + str(epochs))

    # Epochs
    for epoch in range(config.start_epoch, epochs):
        if config.epochs_since_improvement == config.early_stopping:
            break
        if config.epochs_since_improvement > 0 and config.epochs_since_improvement % (config.early_stopping // 2) == 0:
            adjust_learning_rate(optimizers["transformer"], 0.8)
            if config.fine_tune_image_encoder:
                adjust_learning_rate(optimizers["resnet"], 0.8)

        # One epoch's training
        train(config, device, logger,
              train_loader=train_loader,
              image_feature_encoder=image_feature_encoder,
              image_encoder_optimizer=optimizers["resnet"],
              model=model,
              criterion=criterion,
              optimizer=optimizers["transformer"],
              epoch=epoch,
              epochs=epochs)

        # One epoch's validation
        recent_bleu4 = validate(config, device, logger,
                                val_loader=val_loader,
                                image_feature_encoder=image_feature_encoder,
                                model=model,
                                criterion=criterion,
                                epoch=epoch)

        # check if there was an improvement
        is_best = recent_bleu4 > config.best_bleu4
        config.best_bleu4 = max(recent_bleu4, config.best_bleu4)
        if not is_best:
            config.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (config.epochs_since_improvement,))
        else:
            config.epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(config.save_dir, config.model_name, epoch,
                        model, optimizers["transformer"],
                        image_feature_encoder, optimizers["resnet"], is_best)

        if config.dry_run:
            print("Dry run: break epochs")
            break


def train(config, device, logger: SummaryWriter, train_loader, image_feature_encoder, image_encoder_optimizer, model,
          criterion, optimizer,
          epoch, epochs):
    """
    One epoch's training.
    :param train_loader: loader for training data
    :param image_feature_encoder: encoder for the raw input images 
    :param model: model
    :param criterion: label-smoothed cross-entropy loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    image_feature_encoder.train()  # image encoder in eval mode means turning of gradients
    model.train()  # training mode enables dropout

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

    # Batches
    total_batches = len(train_loader)
    for i, (img_before, img_after, target_sequences, target_sequence_lengths, _) in enumerate(train_loader):
        img_before = img_before.to(device)
        img_after = img_after.to(device)
        # Extract Image Features/Image Encoder
        img_before = image_feature_encoder(img_before)  # == source_sequences
        img_after = image_feature_encoder(img_after)  # == source_sequences
        # Move to default device
        image_embedding_attention_heads = config.nb_single_image_attention_heads * 2
        target_sequences = target_sequences.to(device)  # (N, max_target_sequence_pad_length_this_batch)
        source_sequence_lengths = torch.ones(
            len(target_sequence_lengths)) * image_embedding_attention_heads  # source sequence length specifies the number of created visual words = nb_heads
        source_sequence_lengths = source_sequence_lengths.to(device)  # (N)
        target_sequence_lengths = target_sequence_lengths.to(device)  # (N)
        # Time taken to load data
        data_time.update(time.time() - start_data_time)
        # Forward prop.
        predicted_sequences, alpha_bef, alpha_aft = model(img_before, img_after, target_sequences,
                                                          source_sequence_lengths,
                                                          target_sequence_lengths)  # (N, max_target_sequence_pad_length_this_batch, vocab_size)

        # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
        # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
        # Therefore, pads start after (length - 1) positions
        # squeeze the length to dimension (N)
        lengths = target_sequence_lengths.squeeze(1).cpu()
        loss = criterion(inputs=predicted_sequences,
                         targets=target_sequences[:, 1:],
                         lengths=lengths - 1)  # scalar
        loss.backward()
        # Keep track of losses
        losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

        optimizer.step()
        optimizer.zero_grad()

        if image_encoder_optimizer is not None:
            image_encoder_optimizer.step()
        if image_encoder_optimizer is not None:
            image_encoder_optimizer.zero_grad()

        # Update learning rate after each step
        # change_lr(optimizer, new_lr=get_lr(step=step, d_model=d_model, warmup_steps=warmup_steps))
        # Time taken for this training step
        step_time.update(time.time() - start_step_time)
        # Print status
        if (i + 1) % config.print_frequency == 0:
            logger.add_scalar("train/Loss", losses.val, global_step=epoch * total_batches + i)
            logger.add_scalar("train/Loss (avg)", losses.avg, global_step=epoch * total_batches + i)
            print('Epoch {0}/{1}-----'
                  'Batch {2}/{3}-----'
                  'Data Time {data_time.val:.3f} sec ({data_time.avg:.3f}) sec-----'
                  'Step Time {step_time.val:.3f} sec ({step_time.avg:.3f}) sec-----'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch + 1, epochs,
                                                                    i + 1, total_batches,
                                                                    step_time=step_time,
                                                                    data_time=data_time,
                                                                    losses=losses))
        # Reset step time
        start_step_time = time.time()
        # Reset data time
        start_data_time = time.time()

        if config.dry_run:
            print("Dry run: break training")
            break


def validate(config, device, logger: SummaryWriter, val_loader, image_feature_encoder, model, criterion, epoch):
    """
    One epoch's validation.
    :param val_loader: loader for validation data
    :param image_feature_encoder: encoder for raw input
    :param model: model
    :param criterion: label-smoothed cross-entropy loss
    """
    image_feature_encoder.eval()
    model.eval()  # eval mode disables dropout

    references = list()
    hypotheses = list()

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        losses = AverageMeter()
        # Batches 
        for i, (img_before, img_after, target_sequences, target_sequence_lengths, allcaps) in enumerate(
                tqdm(val_loader, total=len(val_loader))):

            # Move to GPU, if available
            img_before = img_before.to(device)
            img_after = img_after.to(device)

            # Extract Image Features/Image Encoder
            img_before = image_feature_encoder(img_before)  # == source_sequences
            img_after = image_feature_encoder(img_after)  # == source_sequences
            image_embedding_attention_heads = config.nb_single_image_attention_heads * 2
            target_sequences = target_sequences.to(device)  # (1, target_sequence_length)
            source_sequence_lengths = torch.ones(
                len(target_sequence_lengths)) * image_embedding_attention_heads  # source sequence length specifies the number of created visual words = nb_heads
            source_sequence_lengths = source_sequence_lengths.to(device)  # (N)
            target_sequence_lengths = target_sequence_lengths.to(device)  # (N)

            # Forward prop.
            predicted_sequence, _, _ = model(img_before, img_after, target_sequences, source_sequence_lengths,
                                             target_sequence_lengths)  # (1, target_sequence_length, vocab_size)

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            lengths = target_sequence_lengths.squeeze(1).cpu()
            loss = criterion(inputs=predicted_sequence,
                             targets=target_sequences[:, 1:],
                             lengths=lengths - 1)  # scalar

            # Keep track of losses
            losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

            # COMPUTE BLEU-4 SCORE for predictions
            # overcome torch cuda tensor handling error => pack the padded sequence into a wrap, more info here: https://discuss.pytorch.org/t/error-with-lengths-in-pack-padded-sequence/35517/12
            # or here: https://github.com/pytorch/pytorch/issues/16542
            torch.set_default_tensor_type(torch.FloatTensor)
            # Remove pad-positions and flatten
            scores_copy = predicted_sequence.clone()
            scores, _, _, _ = pack_padded_sequence(input=predicted_sequence,
                                                   lengths=lengths,
                                                   batch_first=True,
                                                   enforce_sorted=False)  # (sum(lengths), vocab_size)
            if device == torch.device("cuda"):
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

            # References
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {config.word_map['<start>'], config.word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

            if config.dry_run:
                print("Dry run: break validate")
                break

        print("\nValidation loss: %.3f\n\n" % losses.avg)

        # Calculate BLEU-4 Score 
        # Overcome missing overlaps with smoothing, described here: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
        # smoothing_fn = SmoothingFunction().method4
        # bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
        bleu4 = corpus_bleu(references, hypotheses)
        # print(hypotheses)
        print('\n * BLEU-4 - {bleu}\n'.format(bleu=bleu4))
        """
        for refs, h in zip(references[:5], hypotheses[:5]):
            logger.add_text("ref", " ".join(refs[0]), global_step=epoch)
            logger.add_text("hyp", " ".join(h), global_step=epoch)
        """

        logger.add_scalar("validate/Loss (avg)", losses.avg, global_step=epoch)
        logger.add_scalar("validate/BLEU-4", bleu4, global_step=epoch)

        return bleu4

# END OF TRAINING
