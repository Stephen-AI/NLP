# from saber.saber import Saber
from data import QADataset, entity_dict
from typing import List, Dict
from enum import Enum
from json import dump
from utils import save_datatset, load_dataset, split_dataset
import requests
import argparse
import becas

parser = argparse.ArgumentParser()
DATASET_ROOT_PATH = "bioNer-corpora/corpora/NER/CoNLL/BIO"
EMBEDDING_PATH = "embedding/w2v"
NOT_DISEASE_DATASET = "BioNLP13CG_BIO"
DISEASE_DATASET = "NCBI_disease_BIO"

# Training arguments.
parser.add_argument(
    '--train_path',
    type=str,
    required=True,
    help='training dataset path',
)
parser.add_argument(
    '--max_context_length',
    type=int,
    default=384,
    help='maximum context length (do not change!)',
)
parser.add_argument(
    '--max_question_length',
    type=int,
    default=64,
    help='maximum question length (do not change!)',
)


class Tags(Enum):
    SPEC = 0
    ANAT = 1
    DISO = 2
    PATH = 3
    CHED = 4
    ENZY = 5
    MRNA = 6
    PRGE = 7
    COMP = 8
    FUNC = 9
    PROC = 10


#
# param: ent a becas formated entity
# returns: (entity, start, reference str)
def parse_ent(ent: str):
    parts = ent.split("|")
    ref_ent, fmt_concept, start = parts
    fmt_concept = fmt_concept.split(":")
    entity = fmt_concept[3].split(";")[0]
    return entity, int(start), ref_ent


def create_entity_json(start: int, end: int, entity: str):
    return {
        "entity": entity,
        "entity_start": str(start),
        "entity_end": str(end)
    }


def create_string_json(psg_ents: List[Dict], quest_ents: List[Dict]):
    return {
        "passage_entities": psg_ents,
        "question_entities": quest_ents
    }


def get_tags(strs: List[str]):
    pos_to_idx = {0: 0}
    prev_pos = 0
    tags = []
    for i in range(1, len(strs)):
        cur_pos = prev_pos + len(strs[i-1]) + 1
        pos_to_idx[cur_pos] = i
        prev_pos = cur_pos
    sent = " ".join(strs)
    resp = becas.annotate_text(sent)
    for ent in resp['entities']:
        entity, start, ref_ent = parse_ent(ent)
        tags.append(create_entity_json(start, start + len(ref_ent), entity))
    # print(tags)
    return tags


if __name__ == "__main__":
    print(entity_dict)
    # args = parser.parse_args()
    # split_dataset(args.train_path)
    # meta, samples = load_dataset(args.train_path)
    # print(becas.SEMANTIC_GROUPS)
    # text = "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."
    # email = "stephenaigbomian@gmail.com"
    # becas.email = email
    # entities = {}
    # with open('tags.json', 'w') as fi:
    #     for i in range(len(dataset.samples)):
    #         qid, passage, question, answer_start, answer_end = dataset.samples[i]
    #         q_tags = get_tags(question)
    #         p_tags = get_tags(passage)
    #         entities[qid] = create_string_json(p_tags, q_tags)
    #     dump(entities, fi)
    # out_path = "datasets/oherbio.jsonl.gz"
    # save_datatset(out_path, meta, samples)