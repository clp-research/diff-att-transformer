import json
import os
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn
from dotted_dict import DottedDict
import click

import sys

from utils.dataset import TestDataset
from utils.eval_utils import TestContext
from utils.tensorboard import get_summary_writer_for

sys.path.append('.')

from evaluation.eval_transformer_diff_att_clevr_change import perform_eval

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


@click.command()
@click.option("-s", "--data_split", required=True, type=click.Choice(["sc", "nsc"], case_sensitive=True))
@click.argument("rel_output_path", type=str)
def main(data_split, rel_output_path):
    """
    Park perform a "test" run on the whole test data split and store them in two separate files:
    - sc_results.json
    - nsc_results.json

    Their results format is quite simply represented as image_id to caption mappings
    - "001234.png"   : "text caption"         for semantic
    - "001234.png_n" : "text caption"         for non-semantic

    Our test data split does not know the image_id ("001234.png") anymore, but contains the "raw" images.
    Still, we "are sure" that the order is the same as in preprocessing. The image order in preprocessing
    is given by the test set in splits.json and in this particular order the dataloader for the eval should
    return the image pairs. The order is in particular the same for both semantic and non-semantic changes.
    The split only contains integers e.g. 1234 instead of the whole filename. So wee need to convert them.

    Park use an eval script which perform the COCO evaluation on the following test-set results:
    - sc_results.json
    - nsc_results.json
    and they perform a virtual evaluation on the combination of two by combining the file contents into
    - total_results.json

    We create the sc_results.json and nsc_results.json in two separate evaluation runs.
    Once on the sc-only and then on the nsc-only test split.

    The annotation file pre-computed into the COCO evaluation format and saved in the dataset directory as
    - change_captions_reformat.json
    - no_change_captions_reformat.json
    - total_change_captions_reformat.json
    Nevertheless, for the final evaluation run, they only use "total_change_captions_reformat.json"
    """
    split_to_filename = {
        "sc": "sc_results.json",
        "nsc": "nsc_results.json"
    }
    # EVAL TRANSFORMER (DIFF ATTENTION IN IMAGE ENCODER) MODEL
    base_dir = '/home/users/sadler/data'
    data_folder = os.path.join(base_dir, rel_output_path)
    data_name = 'CLEVR_CHANGE_9_cap_per_img_pair_1_min_word_freq'  # base name shared by data files

    # model
    checkpoint_name = "self_att_8_MODEL_FINAL"
    save_dir = "/home/users/sadler/cache/052_block_instruct_transformer/models/clevr_change_sc"
    transformer_checkpoint = save_dir + "/BEST_{}.pth.tar".format(checkpoint_name)

    # Output paths for the particular data_split
    output_dir = "{}/results/clevr_8/selfatt".format(data_folder)
    if not os.path.exists(output_dir):
        raise Exception("Output dir not found at {}\n Please create and try again.".format(output_dir))
    output_file_path = os.path.join(output_dir, split_to_filename[data_split])

    test_split_file_json_path = '/home/users/sadler/data/ImageCorpora/CLEVR_Change/splits.json'
    with open(test_split_file_json_path) as test_file_json:
        test_json_data = json.load(test_file_json)
    image_ids = test_json_data['test']

    # Load word map (word2ix)
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    context = TestContext(output_file_path, word_map, rev_word_map, beam_size=5, nb_heads=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    # Load model
    checkpoint = torch.load(transformer_checkpoint, map_location=torch.device('cuda'))
    model = checkpoint['model'].to(device)
    model.eval()
    image_feature_encoder = checkpoint['image_encoder'].to(device)
    image_feature_encoder.eval()

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = TestDataset(data_folder, data_name, "TEST_" + data_split.upper(),
                             transform=transforms.Compose([normalize]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False)

    # EXECUTE EVALUATION ON TEST DATASET
    print('USING CHECKPOINT: ' + checkpoint_name)
    print('USING DATA SPLIT: ' + "TEST_" + data_split.upper())
    tb_logger = get_summary_writer_for(checkpoint_name)
    perform_eval(data_split, image_ids, context, data_loader, model, image_feature_encoder, device, tb_logger)
    tb_logger.close()


if __name__ == '__main__':
    main()
