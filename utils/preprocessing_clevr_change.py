# IMPORTS
import os
import numpy as np
import h5py
import json
from skimage.transform import resize as imresize
from skimage.io import imread
from skimage import color
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample, shuffle
import sys
import click

sys.path.append('.')


# PREPROCESSING FUNCTIONS
def write_list_to_file(list_to_save, file_path):
    print('...writing list to file.')
    with open(file_path, 'w') as output_file:
        json.dump(list_to_save, output_file)
    print('DONE: writing list to file completed.')


class ImageLocator:

    def __init__(self, default_image_folder, sc_image_folder, nsc_image_folder):
        self.image_folder = {
            "default": default_image_folder,
            "semantic": sc_image_folder,
            "nonsemantic": nsc_image_folder
        }

    def get_path(self, image_id, image_type):
        image_folder = self.image_folder[image_type]
        image_number_filled = str(image_id).zfill(6)
        image_filename = "CLEVR_{}_{}.png".format(image_type, image_number_filled)
        image_path = os.path.join(image_folder, image_filename)
        return image_filename, image_path


class ImageCaptions:

    def __init__(self, change_captions_data, no_change_captions_data):
        self.change_captions_data = change_captions_data
        self.no_change_captions_data = no_change_captions_data

    def get_references(self, image_filename, image_type):
        if image_type == "semantic":
            return self.change_captions_data[image_filename]
        if image_type == "nonsemantic":
            return self.no_change_captions_data[image_filename]
        raise Exception("Unknown type: " + image_type)


def update_image_pairs(image_pairs_captions,
                       image_pairs_paths,
                       img_before_path,
                       img_after_path,
                       reference_captions,
                       word_freq,
                       max_len):
    captions = []
    for single_caption in reference_captions:
        # apply text preprocessing for each caption: lowercasing, tokenization
        image_pair_caption = single_caption.lower().split()
        # Update word frequency
        word_freq.update(image_pair_caption)
        if len(image_pair_caption) <= max_len:
            captions.append(image_pair_caption)
    image_pairs_captions.append(captions)
    image_pairs_paths.append([img_before_path, img_after_path])


def create_split(split_name, annotations, locator: ImageLocator, captions: ImageCaptions,
                 word_freq: Counter, max_len: int, split_mode: str = "both"):
    assert split_mode in ["both", "sc", "nsc"]
    nsc_skip_counter = 0
    sc_skip_counter = 0
    image_pairs_captions = []
    image_pairs_paths = []
    split_set = annotations[split_name]
    for image_id in split_set:
        # each image_id has both a semantic or non-semantic partner
        img_before_filename, img_before_path = locator.get_path(image_id, "default")

        # add the non-semantic change partner
        if split_mode in ["both", "nsc"]:
            img_after_filename, img_after_path = locator.get_path(image_id, "nonsemantic")
            reference_captions = captions.get_references(img_before_filename, "nonsemantic")
            update_image_pairs(image_pairs_captions, image_pairs_paths,
                               img_before_path, img_after_path, reference_captions,
                               word_freq, max_len)
        else:
            nsc_skip_counter = nsc_skip_counter + 1

        # add the semantic change partner
        if split_mode in ["both", "sc"]:
            img_after_filename, img_after_path = locator.get_path(image_id, "semantic")
            reference_captions = captions.get_references(img_before_filename, "semantic")
            update_image_pairs(image_pairs_captions, image_pairs_paths,
                               img_before_path, img_after_path, reference_captions,
                               word_freq, max_len)
        else:
            sc_skip_counter = sc_skip_counter + 1

    print("{} nsc images skipped: {}".format(split_name, nsc_skip_counter))
    print("{}  sc images skipped: {}".format(split_name, sc_skip_counter))
    assert len(image_pairs_captions) == len(image_pairs_paths)
    print("{}: {}".format(split_name, len(image_pairs_captions)))
    if len(image_pairs_captions) == 0:
        raise Exception(split_name + ": No data found")
    return image_pairs_captions, image_pairs_paths


def create_input_files(test_set, split_mode,
                       split_file_json_path,
                       change_captions_json_path,
                       no_change_captions_json_path,
                       default_image_folder,
                       change_image_folder,
                       no_change_image_folder,
                       captions_per_image_pair,
                       min_word_freq,
                       output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    :param split_file_json_path: path of the split file containing the names of the respective file and if it is contained in test, val or train set
    :param change_captions_json_path: path of the file containing the captions for the images with changes
    :param no_change_captions_json_path: path of the file containing the captions for the images with no changes
    :param default_image_folder: folder with downloaded images for training
    :param change_image_folder: folder with downloaded images for validation
    :param no_change_image_folder: folder with downloaded images for testing
    :param captions_per_image_pair: number of captions to sample per image pair
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    print('STARTING preprocessing CLEVR_CHANGE')

    # Storage Variables
    word_freq = Counter()

    with open(split_file_json_path) as split_file_json:
        # contains keys: train, val, test
        # each key points to a list of numbers
        # the numbers refer to the before image in the "images" folder
        split_json_data = json.load(split_file_json)

    with open(change_captions_json_path) as change_captions_json:
        change_captions_data = json.load(change_captions_json)

    with open(no_change_captions_json_path) as no_change_captions_json:
        no_change_captions_data = json.load(no_change_captions_json)

    captions = ImageCaptions(change_captions_data, no_change_captions_data)
    locator = ImageLocator(default_image_folder, change_image_folder, no_change_image_folder)

    if test_set:
        test_image_pairs_captions, test_image_pairs_paths = create_split("test", split_json_data,
                                                                         locator=locator,
                                                                         captions=captions,
                                                                         word_freq=word_freq,
                                                                         max_len=max_len,
                                                                         split_mode=split_mode)
    else:
        train_image_pairs_captions, train_image_pairs_paths = create_split("train", split_json_data,
                                                                           locator=locator,
                                                                           captions=captions,
                                                                           word_freq=word_freq,
                                                                           max_len=max_len,
                                                                           split_mode=split_mode)
        val_image_pairs_captions, val_image_pairs_paths = create_split("val", split_json_data,
                                                                       locator=locator,
                                                                       captions=captions,
                                                                       word_freq=word_freq,
                                                                       max_len=max_len,
                                                                       split_mode=split_mode)

    # Create a base/root name for all output files
    base_filename = "CLEVR_CHANGE_" + str(captions_per_image_pair) + '_cap_per_img_pair_' + str(
        min_word_freq) + '_min_word_freq'

    word_map_file = os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json')
    if test_set:
        # FOR TEST SET ONLY CREATION CASE: use pre-generated wordmap
        # Load word map (word2ix)
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        # No need to dump that file here...
    else:
        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0
        # Save word map to a JSON
        with open(word_map_file, 'w') as j:
            json.dump(word_map, j)

    # Sample captions for each image pair, save image pairs to separate HDF5 file as single image with size (3, 320, 480), and captions and their lengths to JSON files
    seed(123)
    if test_set:
        splits = [(test_image_pairs_paths, test_image_pairs_captions, "TEST_" + split_mode.upper())]
        store(splits, word_map, max_len, output_folder, base_filename)
    else:
        # ommit test set for now, we only want to test on sc data later
        splits = [(train_image_pairs_paths, train_image_pairs_captions, 'TRAIN'),
                  (val_image_pairs_paths, val_image_pairs_captions, 'VAL')]
        store_with_sampling(splits, word_map, captions_per_image_pair, max_len, output_folder, base_filename)


def prepare_image(image_path):
    img_before = imread(image_path)
    img_before_rgb = color.rgba2rgb(img_before)
    img_before = imresize(img_before_rgb, (320, 480))
    assert img_before.shape == (320, 480, 3)
    img_before = img_before.transpose(2, 0, 1)
    assert img_before.shape == (3, 320, 480)
    assert np.max(img_before) <= 255
    return img_before


def store(splits, word_map, max_len, output_folder, base_filename):
    """
        For test split. Store images in hdf5 and captions in JSON.
        Both files are in the same order, so that first image pair corresponds to the first entry in the JSON.
    """
    for image_pairs_paths, image_pairs_captions, split_name in splits:
        # each pair can have multiple captions
        number_of_captions = np.sum([len(image_pairs_captions[idx]) for idx, pair in enumerate(image_pairs_paths)])
        number_of_pairs = len(image_pairs_paths)
        print()
        print("Found", number_of_captions, "captions and", number_of_pairs, "pairs in the split", split_name)
        file_prefix = split_name + '_STORE_' + base_filename
        counter = 0
        with h5py.File(os.path.join(output_folder, file_prefix + '.hdf5'), 'w') as f:
            # Create dataset inside HDF5 file to store images
            storage = f.create_dataset('image_pairs',
                                       (number_of_pairs,
                                        3 * 320 * 480  # before image: channels, height, width
                                        + 3 * 320 * 480  # after image: channels, height, width
                                        ),
                                       dtype='float')
            print("Preprocess '{}' image-pairs and storing to file {}.hdf5".format(number_of_pairs, file_prefix))
            captions = []
            for idx, image_pairs_path in enumerate(tqdm(image_pairs_paths)):
                image_before = prepare_image(image_pairs_path[0])
                image_after = prepare_image(image_pairs_path[1])

                captions_encoded = []
                for caption in image_pairs_captions[idx]:
                    # Encode captions
                    caption_encoded = [word_map['<start>']] \
                                      + [word_map.get(word, word_map['<unk>']) for word in caption] \
                                      + [word_map['<end>']] \
                                      + [word_map['<pad>']] * (max_len - len(caption))
                    captions_encoded.append(caption_encoded)
                captions.append(captions_encoded)
                storage[counter] = np.hstack([image_before.flatten(),  # 3 * 320 * 480 = 460800
                                              image_after.flatten(),  # 3 * 320 * 480 = 460800
                                              ])  # total size per sample = 921600
                counter = counter + 1
        print("Written", counter, "image pairs to", file_prefix + ".hdf5")
        with open(os.path.join(output_folder, file_prefix + '.json'), 'w') as f:
            json.dump(captions, f)
        print("Written", len(captions), "caption lists to", file_prefix + ".json")


def store_with_sampling(splits, word_map, captions_per_image_pair, max_len, output_folder, base_filename):
    """
        For each image-pair we store a number of 'captions_per_image_pair' captions.

        When the number of captions given in the split is higher than 'captions_per_image_pair',
        then we sample a maximum of  'captions_per_image_pair' from the given set.

        When the number of captions given in the split is lower than 'captions_per_image_pair',
        then we re-sample from the given smaller set, so that we reach the amount 'captions_per_image_pair'.
    """
    for impaths, imcaps, split_name in splits:
        with h5py.File(os.path.join(output_folder, split_name + '_IMAGES_BEFORE_' + base_filename + '.hdf5'),
                       'a') as h1, \
                h5py.File(os.path.join(output_folder, split_name + '_IMAGES_AFTER_' + base_filename + '.hdf5'),
                          'a') as h2:
            # Make a note of the number of captions we are sampling per image
            h1.attrs['captions_per_image_pair'] = captions_per_image_pair
            h2.attrs['captions_per_image_pair'] = captions_per_image_pair

            # Create dataset inside HDF5 file to store images
            images_before = h1.create_dataset('images_before', (len(impaths), 3, 320, 480),
                                              dtype='float')  # (channels, height, width)
            images_after = h2.create_dataset('images_after', (len(impaths), 3, 320, 480), dtype='float')

            print("\nReading %s images and captions, storing to file...\n" % split_name)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image_pair:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image_pair - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image_pair)

                # Sanity check
                assert len(captions) == captions_per_image_pair

                # Read image pairs and save them into intended HDF5 file
                img_before = prepare_image(impaths[i][0])
                images_before[i] = img_before

                img_after = prepare_image(impaths[i][1])
                images_after[i] = img_after

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images_before.shape[0] * captions_per_image_pair == len(enc_captions) == len(caplens)
            assert images_after.shape[0] * captions_per_image_pair == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split_name + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split_name + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            # PREPROCESSING EXECUTION


@click.command()
@click.option("-t", "--test_set", is_flag=True)
@click.option("-s", "--split_mode", type=click.Choice(["sc", "nsc", "both"], case_sensitive=True))
@click.argument("rel_output_path", type=str)
def main(test_set, split_mode, rel_output_path):
    # FILE PATHS
    base_dir = '/home/users/sadler/data'  # Add your respective path here
    # split file
    split_file_json_path = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/splits.json')
    default_image_folder = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/images')
    # change captions json path
    change_captions_json_path = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/change_captions.json')
    change_image_folder = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/sc_images')
    # no change captions json path
    no_change_captions_json_path = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/no_change_captions.json')
    no_change_image_folder = os.path.join(base_dir, 'ImageCorpora/CLEVR_Change/nsc_images')
    # output folder to save wordmap and HDF5 file
    # rel_output_path = 'clevr_change_pre'
    output_folder = os.path.join(base_dir, rel_output_path)
    if not os.path.exists(output_folder):
        raise Exception("Output folder does not exist at: " + output_folder)

    create_input_files(test_set, split_mode,
                       split_file_json_path,
                       change_captions_json_path,
                       no_change_captions_json_path,
                       default_image_folder,
                       change_image_folder,
                       no_change_image_folder,
                       captions_per_image_pair=9,
                       min_word_freq=1,
                       output_folder=output_folder,
                       max_len=100)


if __name__ == "__main__":
    main()
