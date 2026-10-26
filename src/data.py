"""Text inputs, graph indexes and batches for the released dataset format."""
import json
import unicodedata
from pathlib import Path
import numpy as np
import torch


def sample_id(row):
    return row.get('sample_id', row.get('id'))


def assert_training_rows(rows):
    if not rows or any(r.get('split') != 'train' for r in rows):
        raise ValueError('Pass only nonempty rows explicitly marked as train.')


def load_rows(dataset_root, task, split):
    if task not in ('admission', 'placement') or split not in ('train', 'valid', 'test'):
        raise ValueError('Unknown task or split.')
    with (Path(dataset_root) / task / (split + '.jsonl')).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if any(sample_id(r) is None for r in rows) or len({sample_id(r) for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate sample IDs.')
    for row in rows:
        if 'split' in row and row['split'] != split:
            raise ValueError('Row split conflicts with its source file.')
        row['split'] = split
    return rows


def load_taxonomy(dataset_root, year=None):
    name = 'static' if year in (None, 0) else str(year - 1)
    return json.loads((Path(dataset_root) / 'taxonomy' / (name + '.json')).read_text())['records']


def legal_candidates(records, row):
    excluded = row['target'] if row.get('masked', False) else None
    return sorted(uid for uid, node in records.items() if uid != excluded
                  and ('trees' not in node or node['trees']))


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def query_tokens(row, tokenizer, max_length=128):
    """Keep serialized CLS/SEP tokens; construct IEEE admission mention windows."""
    if 'query' in row:
        text = row['query']
        serialized = text.lstrip().startswith(tokenizer.cls_token)
        ids = tokenizer.encode(text, add_special_tokens=not serialized,
                               truncation=True, max_length=max_length)
        # Serialized inputs must retain the final separator after truncation.
        if serialized and ids[-1] != tokenizer.sep_token_id:
            ids[-1] = tokenizer.sep_token_id
    else:
        p, start, end = row['paragraph'], int(row['start']), int(row['end'])
        if not 0 <= start < end <= len(p):
            raise ValueError('Invalid mention offsets.')
        mention = p[start:end]
        if row.get('name', mention) != mention:
            raise ValueError('Admission name must equal the observed mention.')
        encode = lambda text: tokenizer.encode(normalize(text), add_special_tokens=False)
        prefix, mt = encode('Query: ' + mention), encode(mention)
        budget = max_length - 3 - len(prefix)
        if not mt or len(mt) > budget:
            raise ValueError('Mention exceeds query token budget.')
        left, right = encode(p[max(0, start-350):start]), encode(p[end:end+350])
        available = budget-len(mt)
        nl = min(len(left), available//2)
        nr = min(len(right), available-nl)
        nl = min(len(left), available-nr)
        ids = [tokenizer.cls_token_id] + prefix + [tokenizer.sep_token_id]
        ids += (left[-nl:] if nl else []) + mt + right[:nr] + [tokenizer.sep_token_id]
    return np.asarray(ids + [tokenizer.pad_token_id]*(max_length-len(ids)), dtype=np.int64)


class Graph:
    def __init__(self, records, tokenizer, candidate_style='auto'):
        self.records = records
        if candidate_style == 'auto':
            candidate_style = 'mesh_temporal' if any('trees' in n for n in records.values()) else 'name_definition'
        self.uids = sorted(u for u, n in records.items() if 'trees' not in n or n['trees'])
        self.index = {u: i for i, u in enumerate(self.uids)}
        allowed = set(self.uids)
        self.parents = {u: set(records[u]['parents']) & allowed for u in self.uids}
        self.children = {u: set() for u in self.uids}
        for u, parents in self.parents.items():
            for p in parents:
                if p not in self.children:
                    raise ValueError('Parent not in candidate graph.')
                self.children[p].add(u)
        if candidate_style == 'mesh_temporal':
            # Original MeSH BE/CE: 128 body tokens + 96 path tokens + CLS/SEP.
            bodies = ['Candidate parent: '+records[u]['name']+'; '+records[u]['definition'] for u in self.uids]
            paths = [' Ancestor paths: '+' | '.join(' > '.join(records[v]['name'] for v in p)
                     for p in records[u]['paths'][:4]) for u in self.uids]
            body = tokenizer(bodies, max_length=128, truncation=True, add_special_tokens=False)['input_ids']
            path = tokenizer(paths, max_length=96, truncation=True, add_special_tokens=False)['input_ids']
            self.tokens = np.zeros((len(self.uids), 226), dtype=np.int64)
            for i, (b, p) in enumerate(zip(body, path)):
                seq = [tokenizer.cls_token_id]+b+p+[tokenizer.sep_token_id]
                self.tokens[i, :len(seq)] = seq
        elif candidate_style == 'ieee_paths':
            # Historical Admission retriever includes the candidate in each path.
            texts = []
            for u in self.uids:
                paths = sorted([records[v]['name'] for v in p+[u]] for p in records[u]['paths'])
                texts.append(records[u]['name']+' [SEP] '+' [SEP] '.join(' > '.join(p) for p in paths))
            self.tokens = np.asarray(tokenizer(texts, max_length=226, padding='max_length',
                                              truncation=True)['input_ids'], dtype=np.int64)
        elif candidate_style == 'name_definition':
            texts = [records[u]['name'] + ' [SEP] ' + records[u].get('definition', '').strip() for u in self.uids]
            self.tokens = np.asarray(tokenizer(texts, max_length=128, padding='max_length',
                                              truncation=True)['input_ids'], dtype=np.int64)
        else:
            raise ValueError('Unknown candidate text style: '+candidate_style)


class Corpus:
    def __init__(self, root, tokenizer, candidate_style='auto'):
        if tokenizer.pad_token_id != 0:
            raise ValueError('The model implementation requires padding token ID 0.')
        self.root, self.tokenizer = Path(root), tokenizer
        self.candidate_style = candidate_style
        self.graphs = {}

    def rows(self, task, split):
        return load_rows(self.root, task, split)

    def graph(self, row):
        year = int(row.get('year', 0))
        if year not in self.graphs:
            self.graphs[year] = Graph(load_taxonomy(self.root, year), self.tokenizer, self.candidate_style)
        return self.graphs[year]

    def queries(self, rows):
        return np.stack([query_tokens(r, self.tokenizer) for r in rows])

    def memory_tokens(self, paths, slots=64):
        if not paths or len(paths) > slots:
            raise ValueError('Expected between one and slots memory paths.')
        ids = self.tokenizer(paths, max_length=226, padding='max_length', truncation=True)['input_ids']
        tokens = np.zeros((slots, 226), dtype=np.int64)
        tokens[:len(ids)] = ids
        roles = np.full(slots, 3, dtype=np.int64)
        roles[:len(ids)] = 0
        return tokens, roles


def admission_batch(examples, device):
    batch = {'query': torch.as_tensor(np.stack([e['query'] for e in examples]), device=device)}
    if 'memory' in examples[0]:
        batch['memory'] = torch.as_tensor(np.stack([e['memory'] for e in examples]), device=device)
        batch['roles'] = torch.as_tensor(np.stack([e['roles'] for e in examples]), device=device)
    if all('label' in e for e in examples):
        batch['labels'] = torch.tensor([e['label'] for e in examples], dtype=torch.float32, device=device)
    return batch
