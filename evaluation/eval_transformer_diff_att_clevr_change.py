import json
import sys

from torch.utils.tensorboard import SummaryWriter

from utils.eval_utils import evaluate, TestContext
from utils.metrics import detokenize_list_of_lists, detokenize_list_of_list_of_lists, \
    formatize_references_and_hypotheses, score

sys.path.append('.')


# CHECKLIST:
# 1. correct model
# 2. correct number of attention heads
# 3. correct test data set loader -> test sets ( nsc test, sc test)
# 4. correct hypothesis objects file path
# 5. correct hypothesis objects file name (png_n)

# sanity check
# assert len(test_image_numbers) == len(hypotheses)
def perform_eval(data_split, image_ids: list, context: TestContext,
                 data_loader, model, image_feature_encoder,
                 device, logger: SummaryWriter):
    references, hypotheses = evaluate(context, data_loader, model, image_feature_encoder, device)

    # Compute metrics
    hypotheses = detokenize_list_of_lists(hypotheses)
    references = detokenize_list_of_list_of_lists(references)

    create_duda_eval_files(data_split, context.output_file_path, image_ids, hypotheses)

    references, hypotheses = formatize_references_and_hypotheses(references, hypotheses)
    scores = score(references, hypotheses)
    print(scores)
    # logger.add_scalars("test/scores_" + data_split, scores)


def image_ids_to_filenames(image_ids, neg=False):
    image_filenames = []
    for image_id in image_ids:
        image_id_str = str(image_id).zfill(6)
        image_filename = image_id_str + '.png'
        if neg:
            image_filename = image_filename + "_n"
        image_filenames.append(image_filename)
    return image_filenames


def create_duda_eval_files(data_split, output_file_path, image_ids, hypotheses):
    """
    Our test data split does not know the image_id ("001234.png") anymore, but contains the "raw" images.

    Still, we "are sure" that the order is the same as in preprocessing. The image order in preprocessing
    is given by the test set in splits.json and in this particular order the dataloader for the eval should
    return the image pairs. The order is in particular the same for both semantic and non-semantic changes.

    The split only contains integers e.g. 1234 instead of the whole filename. So wee need to convert them.
    """
    assert data_split in ["nsc", "sc"]  # the following does not work for "both" (then we need to alternate)!
    image_filenames = image_ids_to_filenames(image_ids, neg=data_split == "nsc")
    hypothesis_objects = [{
        "image_id": image_filename,
        "caption": str(hypothesis)
    } for image_filename, hypothesis in zip(image_filenames, hypotheses)]
    print('Writing hypothesis objects to ' + output_file_path)
    with open(output_file_path, 'w') as f:
        json.dump(hypothesis_objects, f)
