# IMPORTS
# Install evaluation repository for metrics
#pip3 install "git+https://github.com/salaniz/pycocoevalcap.git"
from nltk.tokenize.treebank import TreebankWordDetokenizer
import sys
sys.path.append('.')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


# METRICS

def detokenize_list(list_of_tokens):
  sentence = TreebankWordDetokenizer().detokenize(list_of_tokens)
  return sentence

def detokenize_list_of_lists(list_of_lists):
  new_list = list()
  for token_list in list_of_lists:
    sentence = detokenize_list(token_list)
    new_list.append(sentence)
  return new_list

def detokenize_list_of_list_of_lists(list_of_list_of_lists):
  new_list_of_lists = list()
  for top_list in list_of_list_of_lists:
    bottom_list = detokenize_list_of_lists(top_list)
    new_list_of_lists.append(bottom_list)
  return new_list_of_lists  

def formatize_references_and_hypotheses(references, hypotheses):
    print("The number of references is {}".format(len(references)))
    print("The number of hypotheses is {}".format(len(hypotheses)))
    # return dictionary of idx:hypothesis
    hypotheses = {idx: [hypothesis] for (idx, hypothesis) in enumerate(hypotheses)}
    references = {idx: rr for idx, rr in enumerate(references)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypotheses) != len(references):
        raise ValueError("There is a sentence number mismatch between the inputs")
    return references, hypotheses

def score(references, hypotheses):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(references, hypotheses)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


# INSTRUCTION PARSER
# class that provides crucial functionality to compare generated and gold standard instructions
import time
from collections import Counter

# utility method
def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


def list_match(lst1, lst2):
    if len(lst1) == 0 and len(lst2) == 0:
        return 1.0
    if len(lst1) == 0:
        return 0.0
    inter = intersection(lst1, lst2)
    return len(inter) / len(lst1)


class PSSUnit:
    def __init__(self, text):
        """
        Problem-Solution-Sequence (PSS) Unit
        """
        self.text = text
        self.action = None
        self.target_block_id = -1
        self.target_location = None

        # is True if the PSS was partly generated using the fail safe method,
        # meaning the content of some fields is probably wrong but at least filled
        self.is_fail_safe = False

        # exact landmark ids
        self.landmark_ids = []

        # fine grained features for spatial information
        # e.g. left, right, bottom, etc.
        self.spatial_indicators = []

    def add_spatial_indicator(self, indicator):
        self.spatial_indicators.append(indicator)

    def get_spatial_indicators(self):
        return self.spatial_indicators

    def is_valid(self):
        return not (self.action is None or self.target_block_id is None or self.target_location is None)

    def __str__(self):
        return str(self.__dict__)

    def get_attributes(self):
        return ["action", "target_block_id", "target_location"]

    def compare(self, pss2):
        compared = Counter()
        compared["action"] = float(self.action == pss2.action)
        compared["target_block_id"] = float(self.target_block_id == pss2.target_block_id)
        compared["landmark_ids"] = list_match(self.landmark_ids, pss2.landmark_ids)
        compared["spatial_indicators"] = list_match(self.spatial_indicators, pss2.spatial_indicators)
        return compared


class NumberNormalizer:
    def __init__(self):
        self.numbers_as_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "teen",
                                "eleven", "twelve", "thirteen", "fourteen", "fithteen", "sixteen", "seventeen",
                                "eighteen", "nineteen", "twenty"]

    def get_number_value(self, number_string):
        """
        Returns the int value of a number string if covered
        in the list
        :param number_string:
        :return:
        """
        if number_string in self.numbers_as_word:
            return self.numbers_as_word.index(number_string)
        return -1

    def word_to_number(self, word):
        """
        Parses a word or string number to
        an actual integer
        :param word:
        :return:
        """
        word = word.strip().lower()
        number_index = self.get_number_value(word)
        if number_index == -1:
            composed_number = ''
            start = 0
            for i in range(len(word)):
                if word[start:i] in self.numbers_as_word:
                    composed_number += word[start:i + 1]
                    start = i
            if len(composed_number) > 0:
                return int(self.get_number_value(composed_number))
        return number_index

    def normalize(self, number_string):
        """
        Returns string value of the integer named in the number
        string or None if no number found
        :param number_string:
        :return:
        """
        if number_string.isdigit():
            return number_string
        if number_string.endswith("th") or number_string.endswith("rd") or number_string.endswith("nd"):
            number_string = number_string[:-2]
            if number_string.isdigit(): return number_string
        parsed_number = self.word_to_number(number_string)
        if parsed_number == -1:
            return None
        return str(parsed_number)

class SpatialIndicatorExtractor:
    def __init__(self):
        self.relative_position_terms = ["left", "right", "top", "bottom", "between", "under", "over", "below", "above"]
        self.cardinal_directions = ["north", "east", "south", "west"]

        # add combinations of cardinal directions
        cardinal_dir_comb = []
        for c in self.cardinal_directions:
            for c2 in self.cardinal_directions:
                if c != c2:
                    cardinal_dir_comb.append(c + c2)
        self.cardinal_directions.extend(cardinal_dir_comb)

        self.all_terms = self.relative_position_terms + self.cardinal_directions

    def extract(self, tokens):
        spatial_terms = []
        for token in tokens:
            if token.lemma_ in self.all_terms:
                spatial_terms.append(token.lemma_)
        return spatial_terms


class SimpleInstructionParser:
    def __init__(self, nlp_model=None):
        self.number_normalizer = NumberNormalizer()
        self.spatial_extractor = SpatialIndicatorExtractor()

        self.logo_list = ['adidas', 'bmw', 'burger', 'king', 'coca', 'cola', 'esso', 'heineken', 'hp', 'mcdonalds',
                          'mercedes', 'benz', 'nvidia', 'pepsi', 'shell', 'sri', 'starbucks', 'stella', 'artois',
                          'target', 'texaco', 'toyota', 'twitter', 'ups']
        self.predicate_list = ['reposition', 're-position', 'put', 'place', 'move', 'shift', 'stack', 'connect',
                               'extend', 'lift', 'stick', 'take', 'slide', 'mirror', 'cover', 'drag', 'join', 'nudge',
                               'grab', 'line', 'center', 'rest', 'transfer', 'relocate', 'reorient', 're-orient', 'fit',
                               'push', 'align', 'slid', 'position', 'wedge', 'add', 'moved', 'jump', 'insert', "touch"]

    def get_target_block(self, tokens):
      identified_blocks = list()
      for index, token in enumerate(tokens):
        if token.lower() in self.logo_list:
          identified_blocks.append(token.lower())
      if len(identified_blocks) > 0:
        return identified_blocks[0]
      else:
        return 'default'

    def get_landmarks(self, tokens):
      identified_landmarks = list()
      i = 0
      while i < len(tokens):
        if (tokens[i].lower() in self.logo_list) and (len(identified_landmarks) == 0): # jump over target block and go to index i++
          identified_landmarks.append(tokens[i].lower()) # add target to list to make sure to have seen it
          i +=2
          continue
        else:
          if tokens[i].lower() in self.logo_list:
            identified_landmarks.append(tokens[i].lower())
        i += 1
      if len(identified_landmarks) > 1: # check if next to target landmarks have been found
        return identified_landmarks[1:]
      else:
        return ['default']

    def get_spatial_descriptions(self, tokens):
      spatial_descriptions = list()
      for index, token in enumerate(tokens):
        if token.lower() in self.spatial_extractor.all_terms:
          spatial_descriptions.append(token.lower())
      return spatial_descriptions
    
    def _get_action(self, tokens):
        """
        Extracts action verb from instruction
        :param tokens:
        :return:
        """
        for index, token in enumerate(tokens):
            token_lemma = str(token.lemma_).lower()
            if token_lemma in self.predicate_list or token.pos_ == "VERB":
                return token.text.lower(), index
        return None, index

    def _get_id(self, tokens):
        """
        Extracts block id of the block to be modified
        :param tokens:
        :return:
        """
        for action_index, token in enumerate(tokens):
            number = self.number_normalizer.normalize(token.text)
            if number is not None or token.text.lower() in self.logo_list:
                return number, action_index
        return None, action_index

    def _get_ids(self, tokens):
        """
        Extracts all ids from text
        :param tokens:
        :return:
        """
        ids = []
        for token in tokens:
            number = self.number_normalizer.normalize(token)
            if number is not None:
                ids.append(number)
            elif token.lower() in self.logo_list:
                ids.append(token.lower())
        return ids

    def _get_location(self, tokens, id_index):
        """
         Extracts target location from instruction
        :param tokens:
        :param id_index:
        :return:
        """
        if id_index < len(tokens) - 1:
            location = tokens[id_index + 1:]
            location_string = " ".join([token.text.lower() for token in location])
            return location_string, id_index + 1
        return None, id_index

    def parse_instruction(self, instruction, fail_safe=False):
        """
        Parses the components of an instruction
        :param instruction:
        :param fail_safe: If set to true, will extract components even if none found by
        simply taking the first word as action, second as id and remaining as target_location
        :return:
        """
        tokens = [token for token in self.tokenizer(instruction)]

        # first, try to parse the exact information of the pss unit
        # from the provided tokens
        pss = PSSUnit(instruction)
        pss.action, action_index = self._get_action(tokens)
        pss.target_block_id, id_index = self._get_id(tokens)
        pss.target_location, location_index = self._get_location(tokens, id_index)

        # if not all information could be extracted,
        # fall back to a simple heuristic
        if not pss.is_valid() and fail_safe:
            pss.is_fail_safe = True
            ids = list(self._get_ids([token.text for token in tokens]))

            pss.action = "place"  # default
            pss.target_block_id = ids[0] if len(ids) > 0 else tokens[0].text
            pss.target_location = " ".join([token.text for token in tokens[2:]])

        # add landmark ids
        pss.landmark_ids = self._get_ids([token.text for token in self.tokenizer(pss.target_location)])

        # extract additional spatial terms to indicate spatial overlap between source and generated instruction
        for term in self.spatial_extractor.extract(tokens):
            pss.add_spatial_indicator(term)

        return pss