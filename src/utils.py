from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
from os import path as osp

from tqdm import tqdm
import numpy as np
import bs4
from bs4 import BeautifulSoup as bs
from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize
from lxml import etree
from markuplmft.data.tag_utils import tags_dict


from PIL import Image
from data.html_utils import simplify_dom_tree, rect_fixing, tag_pad_id

logger = logging.getLogger(__name__)


class TIEExample(object):
    r"""
    The Containers for SRC Example.

    Arguments:
        doc_tokens (list[str]): the original tokens of the HTML file before dividing into sub-tokens.
        qas_id (str): the id of the corresponding question.
        html_tree (BeautifulSoup): the Beautiful Soup object created from html code of the corresponding page.
        question_text (str): the text of the corresponding question.
        orig_answer_text (str): the answer text provided by the dataset.
        answer_tid (int): the tid of the answer tag.
        start_position (int): the position where the answer starts in the all_doc_tokens.
        end_position (int): the position where the answer ends in the all_doc_tokens; NOTE that the answer tokens
                            include the token at end_position.
        tok_to_orig_index (list[int]): the mapping from sub-tokens (all_doc_tokens) to original tokens (doc_tokens).
        orig_to_tok_index (list[int]): the mapping from original tokens (doc_tokens) to sub-tokens (all_doc_tokens).
        all_doc_tokens (list[str]): the sub-tokens of the corresponding HTML file.
        tok_to_tags_index (list[int]): the mapping from sub-tokens (all_doc_tokens) to the id of the deepest tag it
                                       belongs to.
        tags_to_tok_index (list[dict]): the starting and ending position (in all_doc_tokens) of each tag.
        orig_tags (list[str]): the list of all tags in the corresponding page in the DFS order.
        tag_depth (list): the depth of each tag in the DOM tree.
        xpath_tag_map (dict): the tag names of the xpath of each tag.
        xpath_subs_map (dict): the subscripts of the xpath of each tag.
    """

    def __init__(self,
                 doc_tokens,
                 qas_id,
                 html_tree=None,
                 question_text=None,
                 orig_answer_text=None,
                 answer_tid=None,
                 start_position=None,
                 end_position=None,
                 tok_to_orig_index=None,
                 orig_to_tok_index=None,
                 all_doc_tokens=None,
                 all_doc_ttypes=None,
                 all_doc_to_bbox=None,
                 image=None,
                 structure_input=None,
                 structure_ttypes=None,
                 structure_to_bbox=None,
                 bbox=None,
                 depth=None,
                 tags_to_tok_index=None,
                 orig_tags=None,):
        self.doc_tokens = doc_tokens
        self.qas_id = qas_id
        self.html_tree = html_tree
        self.question_text = question_text
        self.orig_answer_text = orig_answer_text
        self.answer_tid = answer_tid
        self.start_position = start_position
        self.end_position = end_position
        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.tags_to_tok_index = tags_to_tok_index
        self.orig_tags = orig_tags
        self.all_doc_tokens = all_doc_tokens
        self.all_doc_ttypes = all_doc_ttypes
        self.all_doc_to_bbox = all_doc_to_bbox
        self.image = image
        self.structure_input = structure_input
        self.structure_ttypes = structure_ttypes
        self.structure_to_bbox = structure_to_bbox
        self.bbox = bbox
        self.depth = depth


    def __str__(self):
        return self.__repr__()


class InputFeatures(object):
    r"""
    The Container for the Features of Input Doc Spans.

    Arguments:
        unique_id (int): the unique id of the input doc span.
        example_index (int): the index of the corresponding SRC Example of the input doc span.
        page_id (str): the id of the corresponding web page of the question.
        doc_span_index (int): the index of the doc span among all the doc spans which corresponding to the same SRC
                              Example.
        tokens (list[str]): the sub-tokens of the input sequence, including cls token, sep tokens, and the sub-tokens
                            of the question and HTML file.
        token_to_orig_map (dict[int, int]): the mapping from the HTML file's sub-tokens in the sequence tokens (tokens)
                                            to the original tokens (all_tokens in the corresponding SRC Example).
        token_is_max_context (dict[int, bool]): whether the current doc span contains the max pre- and post-context for
                                                each HTML file's sub-tokens.
        input_ids (list[int]): the ids of the sub-tokens in the input sequence (tokens).
        input_mask (list[int]): use 0/1 to distinguish the input sequence from paddings.
        segment_ids (list[int]): use 0/1 to distinguish the question and the HTML files.
        paragraph_len (int): the length of the HTML file's sub-tokens.
        answer_tid (int): the tid of the answer tag.
        start_position (int): the position where the answer starts in the input sequence (0 if the answer is not fully
                              in the input sequence).
        end_position (int): the position where the answer ends in the input sequence; NOTE that the answer tokens
                            include the token at end_position (0 if the answer is not fully in the input sequence).
        token_to_tag_index (list[int]): the mapping from sub-tokens of the input sequence to the id of the deepest tag
                                        it belongs to.
        tag_to_token_index (list[list[int]]): the starting and ending position (in all_doc_tokens) of each tag.
        app_tags (list[str]): the tid of all tags appearing in the feature.
        base_index (int): the starting position of the content in Structure Encoder.
        is_impossible (bool): whether the answer is fully in the doc span.
        xpath_tags_seq (list): the tag name ids of the xpath of each tag.
        xpath_subs_seq (list): the subscript ids of the xpath of each tag.
    """

    def __init__(self,
                 unique_id,
                 example_index,
                 page_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 tag_ids,
                 input_to_bboxes,
                 bboxes,
                 depths,
                 images,
                 paragraph_len,
                 answer_tid=None,
                 start_position=None,
                 end_position=None,
                 tag_to_token_index=None,
                 app_tags=None,
                 base_index=None,
                 is_impossible=None,):
        self.unique_id = unique_id
        self.page_id = page_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_ids = tag_ids
        self.input_to_bboxes = input_to_bboxes
        self.bboxes = bboxes
        self.depths = depths
        self.images = images
        self.paragraph_len = paragraph_len
        self.answer_tid = answer_tid
        self.start_position = start_position
        self.end_position = end_position
        self.tag_to_token_index = tag_to_token_index
        self.app_tags = app_tags
        self.base_index = base_index
        self.is_impossible = is_impossible

    def to_json(self):
        return json.dumps(self.__dict__)


def html_escape(html):
    r"""
    replace the special expressions in the html file for specific punctuation.
    """
    html = html.replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ')
    return html


def preprocess_html_file_with_rect(html_filename, rect_filename, image_filename):
    image = Image.open(image_filename)
    image = image.convert('RGB')
    with open(rect_filename, "r", encoding='utf-8') as f:
        tag_bbox = json.load(f)

    x_offset = tag_bbox['2']['rect']['x']
    y_offset = tag_bbox['2']['rect']['y']
    for rect in tag_bbox.values():
        rect['rect']['x'] -= x_offset
        rect['rect']['y'] -= y_offset

    with open(html_filename) as f:
        orig_html_string = f.read()

    orig_html = bs4.BeautifulSoup(orig_html_string, 'lxml')

    html = bs4.BeautifulSoup(orig_html_string, 'lxml')
    img_width, img_height = image.size
    simplify_dom_tree(html, tag_bbox, img_width, img_height, less_decompose=True)
    if len(html.find_all()) <= 5 or not html.get_text(strip=True):
        html = orig_html

    rect_fixing(html, tag_bbox, img_width, img_height, orig_html=orig_html)

    final_tag = list(html.children)[0]
    final_tag.name = 'html'

    all_tids = [tag['tid'] for tag in html.find_all()]
    new_tag_bbox = {k: [v['rect']['x'], v['rect']['y'], v['rect']['x'] + v['rect']['width'], v['rect']['y'] + v['rect']['height']]
                    for k, v in tag_bbox.items() if k in all_tids}
    new_tag_bbox[final_tag['tid']] = [0, 0, img_width, img_height]

    return image, str(final_tag), new_tag_bbox


def get_xpath4tokens(html_fn: str, unique_tids: set):
    xpath_map = {}
    tree = etree.parse(html_fn, etree.HTMLParser())
    nodes = tree.xpath('//*')
    for node in nodes:
        tid = node.attrib.get("tid")
        if int(tid) in unique_tids:
            xpath_map[int(tid)] = tree.getpath(node)
    xpath_map[len(nodes)] = "/html"
    xpath_map[len(nodes) + 1] = "/html"
    return xpath_map


def get_xpath_and_treeid4tokens(html_code, unique_tids, max_depth):
    unknown_tag_id = len(tags_dict)
    pad_tag_id = unknown_tag_id + 1
    max_width = 1000
    width_pad_id = 1001

    pad_x_tag_seq = [pad_tag_id] * max_depth
    pad_x_subs_seq = [width_pad_id] * max_depth

    def xpath_soup(element):

        xpath_tags = []
        xpath_subscripts = []
        tree_index = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            para_siblings = parent.find_all(True, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child))

            tree_index.append(next(i for i, s in enumerate(para_siblings, 0) if s is child))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        tree_index.reverse()
        return xpath_tags, xpath_subscripts, tree_index

    xpath_tag_map = {}
    xpath_subs_map = {}

    for tid in unique_tids:
        element = html_code.find(attrs={'tid': tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(element)

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]

        # we do not append them to max depth here

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts

    return xpath_tag_map, xpath_subs_map


def read_examples(input_file, root_dir, is_training, tokenizer, base_mode, feature_extractor, max_depth=50, simplify=False):
    r"""
    pre-process the data in json format into SRC Examples.

    Arguments:
        input_file (str): the inputting data file in json format.
        root_dir (str): the root directory of the raw WebSRC dataset, which contains the HTML files.
        is_training (bool): True if processing the training set, else False.
        tokenizer (Tokenizer): the tokenizer for PLM in use.
        base_mode (str): the name of the base model of Content Encoder.
        max_depth (int): the maximum depth limit used for xpath embedding generation.
        simplify (bool): when setting to Ture, the returned Example will only contain document tokens, the id of the
                         question-answers, the Beautiful Soup object of the html code, and the depth of all tags.
    Returns:
        list[SRCExamples]: the resulting SRC Examples, contained all the needed information for the feature generation
                           process, except when the argument simplify is setting to True;
        set[str]: all the tag names appeared in the processed dataset, e.g. <div>, <img/>, </p>, etc..
    """
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def html_to_text_list(h):
        tag_num, text_list = 0, []
        for element in h.descendants:
            if (type(element) == bs4.element.NavigableString) and (element.strip()):
                text_list.append(element.strip())
            if type(element) == bs4.element.Tag:
                tag_num += 1
        return text_list, tag_num + 2  # + 2 because we treat the additional 'yes' and 'no' as two special tags.

    def html_to_text(h):
        tag_list = set()
        for element in h.descendants:
            if type(element) == bs4.element.Tag:
                element.attrs = {}
                temp = str(element).split()
                tag_list.add(temp[0])
                tag_list.add(temp[-1])
        return html_escape(str(h)), tag_list

    def adjust_offset(offset, text):
        text_list = text.split()
        cnt, adjustment = 0, []
        for t in text_list:
            if not t:
                continue
            if t[0] == '<' and t[-1] == '>':
                adjustment.append(offset.index(cnt))
            else:
                cnt += 1
        add = 0
        adjustment.append(len(offset))
        for i in range(len(offset)):
            while i >= adjustment[add]:
                add += 1
            offset[i] += add
        return offset

    def e_id_to_t_id(e_id, html):
        t_id, real_eid = 0, 0
        for element in html.descendants:
            if type(element) == bs4.element.NavigableString and element.strip():
                t_id += 1
            if type(element) == bs4.element.Tag:
                if int(element.attrs['tid']) >= e_id:
                    break
                else:
                    real_eid += 1
        return t_id, real_eid

    def calc_num_from_raw_text_list(t_id, l):
        n_char = 0
        for i in range(t_id):
            n_char += len(l[i]) + 1
        return n_char

    def get_html_tag_num(h):
        tag_num = 0
        for element in h.descendants:
            if type(element) == bs4.element.Tag:
                tag_num += 1
        return tag_num + 2

    def word_to_tag_from_text(tokens, h):
        cnt, path = -1, []
        w2t, t2w, tags = [], [], []
        for ind in range(len(tokens) - 2):
            t = tokens[ind]
            if len(t) < 2:
                w2t.append(path[-1])
                continue
            if t[0] == '<' and t[-2] == '/':
                cnt += 1
                w2t.append(cnt)
                tags.append(t)
                t2w.append({'start': ind, 'end': ind})
                continue
            if t[0] == '<' and t[1] != '/':
                cnt += 1
                path.append(cnt)
                tags.append(t)
                t2w.append({'start': ind})
            w2t.append(path[-1])
            if t[0] == '<' and t[1] == '/':
                num = path.pop()
                t2w[num]['end'] = ind
        w2t.append(cnt + 1)
        w2t.append(cnt + 2)
        tags.append('<no>')
        tags.append('<yes>')
        t2w.append({'start': len(tokens) - 2, 'end': len(tokens) - 2})
        t2w.append({'start': len(tokens) - 1, 'end': len(tokens) - 1})
        assert len(w2t) == len(tokens)
        assert len(tags) == len(t2w), (len(tags), len(t2w))
        assert len(path) == 0, h
        return w2t, t2w, tags

    def word_tag_offset(html):
        w_t, t_w, tags, tags_tids = [], [], [], []
        for element in html.descendants:
            if type(element) == bs4.element.Tag:
                content = ' '.join(list(element.strings)).split()
                t_w.append({'start': len(w_t), 'len': len(content)})
                tags.append('<' + element.name + '>')
                tags_tids.append(int(element['tid']))
            elif type(element) == bs4.element.NavigableString and element.strip():
                text = element.split()
                tid = int(element.parent['tid'])
                ind = tags_tids.index(tid)
                for _ in text:
                    w_t.append(ind)
        t_w.append({'start': len(w_t), 'len': 1})
        t_w.append({'start': len(w_t) + 1, 'len': 1})
        w_t.append(len(tags))
        w_t.append(len(tags) + 1)
        tags.append('<no>')
        tags.append('<yes>')
        tags_tids.append(-2)
        tags_tids.append(-1)
        return w_t, t_w, tags, tags_tids

    def subtoken_tag_offset(html, s_tok, tok_s):
        w_t, t_w, tags, tags_tids = word_tag_offset(html)
        s_t, t_s = [], []
        unique_tids = set(range(len(tags)))
        for i in range(len(s_tok)):
            s_t.append(w_t[s_tok[i]])
        for i in t_w:
            try:
                t_s.append({'start': tok_s[i['start']], 'end': tok_s[i['start'] + i['len']] - 1})
            except IndexError:
                assert i == t_w[-1]
                t_s.append({'start': tok_s[i['start']], 'end': len(s_tok) - 1})
        return s_t, t_s, tags_tids, unique_tids

    def calculate_depth(html_code):
        def _calc_depth(tag, depth):
            for t in tag.contents:
                if type(t) != bs4.element.Tag:
                    continue
                tag_depth.append(depth)
                _calc_depth(t, depth + 1)

        tag_depth = []
        _calc_depth(html_code, 1)
        tag_depth += [2, 2]
        return tag_depth

    examples = []
    all_tag_list = set()
    for entry in input_data:
        domain = entry["domain"]
        for website in entry["websites"]:

            # Generate Doc Tokens
            page_id = website["page_id"]
            curr_dir = osp.join(root_dir, domain, page_id[0:2], 'processed_data')

            html_filename = osp.join(curr_dir, page_id + '.html')
            rect_filename = osp.join(curr_dir, page_id + '.json')
            image_filename = osp.join(curr_dir, page_id + '.png')

            image, simplified_html_string, tag_bbox = \
                preprocess_html_file_with_rect(html_filename, rect_filename, image_filename)

            feature = feature_extractor(image, simplified_html_string, tag_bbox, return_tids=True,
                                        return_depth=True)

            image = feature['pixel_values'][0]
            structure_input = feature['structure_inputs'][0]
            structure_ttype = feature['structure_ttypes'][0]
            structure2bbox = feature['structure2bboxes'][0]

            content_input = feature['content_inputs'][0]
            content_ttype = feature['content_ttypes'][0]
            content2bbox = feature['content2bboxes'][0]

            bbox = feature['bboxes'][0]
            depth = feature['bbox_depths'][0]
            bbox_tid = feature['bbox_tids'][0]

            orig_html_file = open(html_filename).read()
            orig_html = bs(orig_html_file, "html.parser")

            html_code = bs(simplified_html_string, "html.parser")

            char_to_span_idx = []
            page_text = ''
            for i, text_span in enumerate(content_input):
                if page_text != "":
                    page_text += ' '
                    char_to_span_idx += [i - 1]
                page_text += text_span
                char_to_span_idx += [i] * len(text_span)

            doc_tokens = []
            doc_ttypes = []
            doc_to_tag_index = []
            char_to_word_offset = []
            doc_to_bbox = []

            page_text = ' '.join(content_input)
            prev_is_whitespace = True
            for i, c in enumerate(page_text):
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                        span_idx = char_to_span_idx[i]
                        tid = int(bbox_tid[content2bbox[span_idx]])
                        doc_to_tag_index.append(tid)
                        doc_ttypes.append(content_ttype[span_idx])
                        doc_to_bbox.append(content2bbox[span_idx])
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            tag_num = get_html_tag_num(orig_html)

            doc_tokens.append('no')
            char_to_word_offset.append(len(doc_tokens) - 1)
            doc_ttypes.append(122)
            doc_to_tag_index.append(tag_num - 2)
            doc_to_bbox.append(structure2bbox[0])

            doc_tokens.append('yes')
            char_to_word_offset.append(len(doc_tokens) - 1)
            doc_ttypes.append(122)
            doc_to_tag_index.append(tag_num - 1)
            doc_to_bbox.append(structure2bbox[0])

            tag_list = []

            assert len(doc_tokens) == char_to_word_offset[-1] + 1, (len(doc_tokens), char_to_word_offset[-1])

            if simplify:
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    example = TIEExample(doc_tokens=doc_tokens, qas_id=qas_id, html_tree=html_code)
                    examples.append(example)
            else:
                # Tokenize all doc tokens
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                all_doc_ttypes = []
                all_doc_to_bbox = []
                tok_to_tags_index = []
                tags_to_tok_index = []
                orig_tags = []

                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    if token in tag_list:
                        sub_tokens = [token]
                    else:
                        sub_tokens = tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)
                        all_doc_ttypes.append(doc_ttypes[i])
                        all_doc_to_bbox.append(doc_to_bbox[i])
                        tok_to_tags_index.append(doc_to_tag_index[i])

                tok_to_tags_index, tags_to_tok_index, orig_tags, unique_tids = subtoken_tag_offset(
                                                                                        html_code,
                                                                                        tok_to_orig_index,
                                                                                        orig_to_tok_index)

                # assert tok_to_tags_index[-1] == tag_num - 1, (tok_to_tags_index[-1], tag_num - 1)

                # Process each qas, which is mainly calculate the answer position
                for qa in website["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    answer_tag_idx = None
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    # tag_depth = calculate_depth(html_code)

                    if is_training:
                        if len(qa["answers"]) != 1:
                            raise ValueError("For training, each question should have exactly 1 answer.")
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        if answer["element_id"] == -1:
                            answer_tag_idx = len(orig_tags) - 2 + answer["answer_start"]
                            num_char = len(char_to_word_offset) - 2
                        else:
                            num_text, answer_tag_idx = e_id_to_t_id(answer["element_id"], html_code)
                            num_char = calc_num_from_raw_text_list(num_text, content_input)
                        answer_offset = num_char + answer["answer_start"]
                        answer_length = len(orig_answer_text) if answer["element_id"] != -1 else 1
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        node_text = doc_tokens[tok_to_orig_index[tags_to_tok_index[answer_tag_idx]['start']]:
                                               tok_to_orig_index[tags_to_tok_index[answer_tag_idx]['end']] + 1]
                        node_text = ' '.join([s for s in node_text if s[0] != '<' or s[-1] != '>'])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if node_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                           qa['id'], node_text, cleaned_answer_text)
                            continue
                        actual_text = " ".join([w for w in doc_tokens[start_position:(end_position + 1)]
                                                if w[0] != '<' or w[-1] != '>'])
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                           qa['id'], actual_text, cleaned_answer_text)
                            continue

                    example = TIEExample(
                        doc_tokens=doc_tokens,
                        qas_id=qas_id,
                        html_tree=html_code,
                        question_text=question_text,
                        orig_answer_text=orig_answer_text,
                        answer_tid=orig_tags[answer_tag_idx] if answer_tag_idx is not None else None,
                        start_position=start_position,
                        end_position=end_position,
                        tok_to_orig_index=tok_to_orig_index,
                        orig_to_tok_index=orig_to_tok_index,
                        all_doc_tokens=all_doc_tokens,
                        all_doc_ttypes=all_doc_ttypes,
                        all_doc_to_bbox=all_doc_to_bbox,
                        image=image,
                        structure_input=structure_input,
                        structure_ttypes=structure_ttype,
                        structure_to_bbox=structure2bbox,
                        bbox=bbox,
                        depth=depth,
                        tags_to_tok_index=tags_to_tok_index,
                        orig_tags=orig_tags,
                    )
                    examples.append(example)

    return examples, all_tag_list


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_tag_length,
                                 doc_stride, max_query_length, max_structure_length, is_training,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 max_depth=50):
    r"""
    Converting the SRC Examples further into the features for all the input doc spans.

    Arguments:
        examples (list[SRCExample]): the list of SRC Examples to process.
        tokenizer (Tokenizer): the tokenizer for PLM in use.
        max_seq_length (int): the max length of the total sub-token sequence, including the question, cls token, sep
                              tokens, and contents; if the length of the input is bigger than max_seq_length, the input
                              will be cut into several doc spans.
        max_tag_length (ing): the max length of the total tag sequence, including the question, cle token, sep tokens,
                              and the html tags corresponding to the content; if the length of the input is bigger than
                              max_tag_length, an error will occur.
        doc_stride (int): the stride length when the input is cut into several doc spans.
        max_query_length (int): the max length of the sub-token sequence of the questions; the question will be truncate
                                if it is longer than max_query_length.
        is_training (bool): True if processing the training set, else False.
        cls_token (str): the cls token in use, default is '[CLS]'.
        sep_token (str): the sep token in use, default is '[SEP]'.
        pad_token (int): the id of the padding token in use when the total sub-token length is smaller that
                         max_seq_length, default is 0 which corresponding to the '[PAD]' token.
        sequence_a_segment_id (int): the segment id for the first sequence (the question), default is 0.
        sequence_b_segment_id (int): the segment id for the second sequence (the html file), default is 1.
        cls_token_segment_id (int): the segment id for the cls token, default is 0.
        pad_token_segment_id (int): the segment id for the padding tokens, default is 0.
        max_depth (int): the maximum depth limit used for xpath embedding generation.
    Returns:
        list[InputFeatures]: the resulting input features for all the input doc spans
    """

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(tqdm(examples)):

        structure_tokens = example.structure_input
        structure_ttype = example.structure_ttypes
        structure_to_bbox = example.structure_to_bbox
        if len(structure_tokens) > max_structure_length:
            structure_tokens = structure_tokens[0:max_structure_length]
            structure_ttype = structure_ttype[0:max_structure_length]
            structure_to_bbox = structure_to_bbox[0:max_structure_length]

        bboxes = example.bbox
        depths = example.depth
        bboxes += [[0, 0, 0, 0]] * (max_seq_length - len(bboxes))
        depths += [0] * (max_seq_length - len(depths))
        image = example.image

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_answer_tid = None
        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_answer_tid = example.answer_tid
            tok_start_position = example.orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = example.orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(example.all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                example.all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - len(structure_tokens) - 5

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(example.all_doc_tokens):
            length = len(example.all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            assert length > 0, (max_tokens_for_doc, len(example.all_doc_tokens), start_offset)
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(example.all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}

            tag_ids = []
            input_to_bboxes = []

            tag_to_token_index = []

            # CLS
            tokens.append(tokenizer.cls_token)
            tag_to_token_index.append([0, 0])
            tag_ids.append(56)
            input_to_bboxes.append(structure_to_bbox[0])

            for i in range(len(structure_tokens)):
                tag_to_token_index.append([len(tokens) + i, len(tokens) + i])
            tokens += structure_tokens
            tag_ids += structure_ttype
            input_to_bboxes += structure_to_bbox

            # SEP token
            tokens += [tokenizer.sep_token] * 2
            tag_ids += [56] * 2
            input_to_bboxes += [structure_to_bbox[0]] * 2
            tag_to_token_index.append([len(tokens) - 2, len(tokens) - 2])
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])

            segment_ids = [0] * len(tokens)

            # Query
            for i in range(len(query_tokens)):
                tag_to_token_index.append([len(tokens) + i, len(tokens) + i])
            tokens += query_tokens
            segment_ids += [1] * len(query_tokens)
            tag_ids += [56] * len(query_tokens)
            input_to_bboxes += [structure_to_bbox[0]] * len(query_tokens)

            # SEP token
            tokens.append(tokenizer.sep_token)
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
            segment_ids.append(1)
            tag_ids.append(56)
            input_to_bboxes.append(structure_to_bbox[0])

            # Paragraph
            app_tags = []
            for i in range(len(example.tags_to_tok_index)):
                start = example.tags_to_tok_index[i]['start']
                end = example.tags_to_tok_index[i]['end']
                if end < doc_span.start:
                    continue
                elif start >= doc_span.start + doc_span.length:
                    continue
                elif start > end:
                    continue
                else:
                    start = max(start, doc_span.start) - doc_span.start + len(tokens)
                    end = min(end, doc_span.start + doc_span.length - 1) - doc_span.start + len(tokens)
                    tag_to_token_index.append([start, end])
                    app_tags.append(example.orig_tags[i])

            if len(app_tags) > max_tag_length - len(query_tokens) - len(structure_tokens) - 5:
                raise ValueError('Max tag length is not big enough to contain {}'.format(len(app_tags)))
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = example.tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(example.all_doc_tokens[split_token_index])
                segment_ids.append(1)
                tag_ids.append(example.all_doc_ttypes[split_token_index])
                input_to_bboxes.append(example.all_doc_to_bbox[split_token_index])

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(1)
            tag_to_token_index.append([len(tokens) - 1, len(tokens) - 1])
            tag_ids.append(56)
            input_to_bboxes.append(structure_to_bbox[0])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # mask generating
            base = len(query_tokens) + len(structure_tokens) + 4
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0)
                segment_ids.append(0)
                tag_ids.append(tag_pad_id)
                input_to_bboxes.append(max_seq_length - 1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = False
            answer_tid = None
            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    answer_tid = 0
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    assert tok_answer_tid in app_tags
                    offset = len(query_tokens) + len(structure_tokens) + 4
                    answer_tid = app_tags.index(tok_answer_tid) + offset
                    start_position = tok_start_position - doc_start + offset
                    end_position = tok_end_position - doc_start + offset

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    page_id=example.qas_id[:-5],
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    tag_ids=tag_ids,
                    input_to_bboxes=input_to_bboxes,
                    bboxes=bboxes,
                    depths=depths,
                    images=image,
                    paragraph_len=paragraph_len,
                    answer_tid=answer_tid,
                    start_position=start_position,
                    end_position=end_position,
                    tag_to_token_index=tag_to_token_index,
                    app_tags=app_tags,
                    base_index=base,
                    is_impossible=span_is_impossible,
                ))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawTagResult = collections.namedtuple("RawResult", ["unique_id", "tag_logits"])


def write_tag_predictions(all_examples, all_features, all_results, n_best_tag_size, model_name,
                          output_tag_prediction_file, output_nbest_file, write_pred):
    r"""
    Compute and write down the final answer tag prediction results.

    Arguments:
        all_examples (list[SRCExample]): all the SRC Example of the dataset; note that we only need it to provide the
                                         mapping from example index to the question-answers id and the original tag
                                         list.
        all_features (list[InputFeatures]): all the features for the input doc spans.
        all_results (list[RawResult]): all the results from the models.
        n_best_tag_size (int): the number of the n best buffer and the final n best result saved.
        model_name (str): the name of the model used for Content Encoder.
        output_tag_prediction_file (str): the file which the best answer tag predictions will be written to.
        output_nbest_file (str): the file which the n best answer tag predictions including text, tag, and probabilities
                                 will be written to.
        write_pred (bool): whether to write the predictions to the disk.
    Returns:
        dict: the n best answer prediction results.
        dict: the best answer tag prediction results.
    """
    logger.info("Writing tag predictions to: %s" % output_tag_prediction_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "tag_index", "tag_logit", "tag_id"])

    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            possible_values = [ind for ind in range(feature.base_index,
                                                    feature.base_index + len(feature.app_tags))]
            if model_name != 'markuplm':
                possible_values = [ind for ind in possible_values
                                   if feature.tag_to_token_index[ind][1]
                                   - feature.tag_to_token_index[ind][0] > 1 or
                                   example.orig_tags.index(feature.app_tags[ind - feature.base_index]) in ['<no>', '<yes>']]
            tag_indexes = _get_best_tags(result.tag_logits, n_best_tag_size, possible_values)
            for ind in range(len(tag_indexes)):
                tag_index = tag_indexes[ind]
                if tag_index == 0:
                    continue
                tag_logit = result.tag_logits[tag_index]
                tag_id = feature.app_tags[tag_index - feature.base_index]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        tag_index=tag_index,
                        tag_logit=tag_logit,
                        tag_id=tag_id))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.tag_logit,
            reverse=True)

        _NBestPrediction = collections.namedtuple(
            "NBestPrediction", ["tag_logit", "tag_id"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_tag_size:
                break
            if pred.tag_index > 0:  # this is a non-null prediction
                if pred.tag_id in seen_predictions:
                    continue
                seen_predictions[pred.tag_id] = True
            else:
                seen_predictions[-1] = True

            nbest.append(_NBestPrediction(tag_logit=pred.tag_logit, tag_id=pred.tag_id))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NBestPrediction(tag_logit=0.0, tag_id=-1))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.tag_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["probability"] = probs[i]
            output["tag_logit"] = entry.tag_logit
            output["tag_id"] = entry.tag_id
            nbest_json.append(output)
        assert len(nbest_json) >= 1

        best_tag = nbest_json[0]["tag_id"]
        all_tag_predictions[example.qas_id] = best_tag
        all_nbest_json[example.qas_id] = nbest_json

    if write_pred:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        with open(output_tag_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_tag_predictions, indent=4) + '\n')

    return all_nbest_json, all_tag_predictions


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def write_predictions_provided_tag(all_examples, all_features, all_results, n_best_size, max_answer_length,
                                   do_lower_case, output_prediction_file, input_tag_prediction_file,
                                   output_refined_tag_prediction_file, output_nbest_file, verbose_logging,
                                   write_pred):
    r"""
    Providing the n best answer tag predictions, compute and write down the final answer span prediction results,
    including the n best results.

    Arguments:
        all_examples (list[SRCExample]): all the SRC Example of the dataset; note that we only need it to provide the
                                         mapping from example index to the question-answers id.
        all_features (list[InputFeatures]): all the features for the input doc spans.
        all_results (list[RawResult]): all the results from the models.
        n_best_size (int): the number of the n best buffer and the final n best result saved.
        max_answer_length (int): constrain the model to predict the answer no longer than it.
        do_lower_case (bool): whether the model distinguish upper and lower case of the letters.
        output_prediction_file (str): the file which the best answer text predictions will be written to.
        input_tag_prediction_file (str/dict): the file which the n best answer tag predictions has been written to, or
                                              the n best answer tag prediction results.
        output_refined_tag_prediction_file (str): the file which the refined best answer tag predictions will be written
                                                  to.
        output_nbest_file (str): the file which the n best answer predictions including text, tag, and probabilities
                                 will be written to.
        verbose_logging (bool): if true, all the warnings related to data processing will be printed.
        write_pred (bool): whether to write the predictions to the disk.
    Return:
        dict: the best answer span prediction results.
        dict: the refined best answer tag prediction results.
    """
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    def _get_tag_id(ind2tok, start_ind, end_ind, base, ind2tag):
        tag_ind = -1
        for ind in range(base, len(ind2tok)):
            if (start_ind >= ind2tok[ind][0]) and (end_ind <= ind2tok[ind][1]):
                tag_ind = ind
        tag_ind -= base
        assert tag_ind >= 0
        return ind2tag[tag_ind]

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index",
         "tag_index", "start_index", "end_index",
         "tag_logit", "start_logit", "end_logit",
         "tag_id"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_refined_tag_predictions = collections.OrderedDict()
    if isinstance(input_tag_prediction_file, str):
        all_tag_predictions = json.load(open(input_tag_prediction_file))
    else:
        all_tag_predictions = input_tag_prediction_file

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        nb_tag_pred = all_tag_predictions[example.qas_id]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            for item in nb_tag_pred:
                tag_pred = item['tag_id']
                if tag_pred not in feature.app_tags:
                    continue
                tag_index = feature.app_tags.index(tag_pred) + feature.base_index
                left_bound, right_bound = feature.tag_to_token_index[tag_index]
                start_indexes = _get_best_indexes(result.start_logits[left_bound:right_bound + 1], n_best_size)
                end_indexes = _get_best_indexes(result.end_logits[left_bound:right_bound + 1], n_best_size)
                start_indexes = [ind + left_bound for ind in start_indexes]
                end_indexes = [ind + left_bound for ind in end_indexes]
                tag_logit = item['tag_logit']
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        tag_ids = _get_tag_id(feature.tag_to_token_index,
                                              start_index, end_index,
                                              feature.base_index, feature.app_tags)
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                tag_index=tag_index,
                                start_index=start_index,
                                end_index=end_index,
                                tag_logit=tag_logit,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                                tag_id=tag_ids))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NBestPrediction = collections.namedtuple(
            "NBestPrediction", ["text", "tag_logit", "start_logit", "end_logit", "tag_id"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if '{} with the tag_id of {}'.format(final_text, str(pred.tag_index)) in seen_predictions:
                continue
            seen_predictions['{} with the tag_id of {}'.format(final_text, str(pred.tag_index))] = True

            nbest.append(
                _NBestPrediction(
                    text=final_text,
                    tag_logit=pred.tag_logit,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    tag_id=pred.tag_id))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NBestPrediction(
                    text='empty',
                    start_logit=0.0,
                    end_logit=0.0,
                    tag_logit=0.0,
                    tag_id=-1))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["tag_logit"] = entry.tag_logit
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["tag_id"] = entry.tag_id
            nbest_json.append(output)
        assert len(nbest_json) >= 1

        best_text = nbest_json[0]["text"].split()
        best_text = ' '.join([w for w in best_text if w[0] != '<' or w[-1] != '>'])
        all_predictions[example.qas_id] = best_text
        best_tag = nbest_json[0]["tag_id"]
        all_refined_tag_predictions[example.qas_id] = best_tag
        all_nbest_json[example.qas_id] = nbest_json

    if write_pred:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        with open(output_refined_tag_prediction_file, 'w') as writer:
            writer.write(json.dumps(all_refined_tag_predictions, indent=4) + '\n')

    return all_predictions, all_refined_tag_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _get_best_tags(logits, n_best_size, possible_values):
    """Get the n-best logits from a list with exclusions."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if index_and_score[i][0] not in possible_values:
            continue
        if len(best_indexes) >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def form_dom_mask(app, tree, accelerate=True):
    r"""
    Generate DOM mask

    Arguments:
        app (list): the tags which appears in the feature.
        tree (BeautifulSoup): the Beautiful Soup object of the web page.
        accelerate (bool): whether to use the accelerated DOM tree (if true) or the original DOM tree (if false).
    Returns:
        numpy.array: the generated DOM mask.
        numpy.array: the directed parent-child relation mask.
    """
    def _unit(node):
        for ch in node.contents:
            if type(ch) != bs4.element.Tag:
                continue
            curr = int(ch['tid'])
            if curr not in app:
                continue
            curr = app.index(curr)
            if len(path) > 0:
                children[path[-1], curr] = 1
            if accelerate:
                for par in path:
                    adj[par, curr] = 1
                    adj[curr, par] = 1
            else:
                if len(path) > 0:
                    adj[path[-1], curr] = 1
                    adj[curr, path[-1]] = 1
            path.append(curr)
            _unit(ch)
            path.pop()

    path = []
    adj = np.zeros((len(app), len(app)), dtype=np.int)
    children = np.zeros((len(app), len(app)), dtype=np.int)
    ind = np.diag_indices_from(adj)
    adj[0] = 1
    adj[:, 0] = 1
    adj[ind] = 1
    _unit(tree)
    return adj, children


def form_npr_mask(app, rel, direction='B'):
    r"""
    Generate NPR mask

    Arguments:
        app (list): the tags which appears in the feature.
        rel (dict{str: dict}): the pre-generated information for NPR mask generation.
        direction (str): the relations used in the NPR graph.
    Returns:
        numpy.array: the generated NPR mask.
    """
    def _form_direction_mask(rel, d):
        mask = np.zeros((len(app), len(app)), dtype=np.int)
        reverse_mask = np.zeros((len(app), len(app)), dtype=np.int)
        ind = np.diag_indices_from(mask)
        mask[ind] = 1
        for k, v in rel[d].items():
            try:
                curr = app.index(int(k))
            except ValueError:
                continue
            for t in v:
                try:
                    ter = app.index(int(t))
                except ValueError:
                    continue
                mask[curr, ter] = 1
                reverse_mask[ter, curr] = 1
        return mask, reverse_mask
    o = []
    if direction in ['B', 'H']:
        l, rl = _form_direction_mask(rel, 'left')
        r, rr = _form_direction_mask(rel, 'right')
        l[rr == 1] = 1
        r[rl == 1] = 1
        o += [r, l]
    if direction in ['B', 'V']:
        u, ru = _form_direction_mask(rel, 'up')
        d, rd = _form_direction_mask(rel, 'down')
        u[rd == 1] = 1
        d[ru == 1] = 1
        o += [u, d]
    return np.stack(o)
