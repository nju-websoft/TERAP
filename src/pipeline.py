from .configuration import input_settings
"""Batched retrieval, frozen Placement features and end-to-end prediction."""
from collections import defaultdict
import numpy as np
import torch
from .data import admission_batch, sample_id, assert_training_rows
from .memory import build_memory
from .retrieval import pair


def device_of(model):
    return next(model.parameters()).device


class RetrievalEngine:
    """In-memory parent embeddings; clear_cache() after changing BE weights."""
    def __init__(self, corpus, be, ce=None, batch_size=128):
        self.corpus, self.be, self.ce = corpus, be, ce
        if ce is not None and corpus.candidate_style == 'ieee_paths':
            ce.inference_fp32 = True
        self.batch_size = batch_size
        self.banks = {}

    def clear_cache(self):
        self.banks.clear()

    @torch.inference_mode()
    def bank(self, row):
        year = int(row.get('year', 0))
        if year not in self.banks:
            graph = self.corpus.graph(row)
            self.be.eval()
            parts = [self.be.encode(torch.as_tensor(graph.tokens[i:i+self.batch_size],
                       device=device_of(self.be)), 'parent').cpu()
                     for i in range(0, len(graph.uids), self.batch_size)]
            self.banks[year] = torch.cat(parts)
        return self.banks[year]

    @torch.inference_mode()
    def score_pairs(self, row, candidates):
        if self.ce is None or not candidates:
            raise ValueError('CE and nonempty candidate UIDs are required.')
        graph = self.corpus.graph(row)
        query = self.corpus.queries([row])[0]
        self.ce.eval()
        scores, features = [], []
        for start in range(0, len(candidates), self.batch_size):
            uids = candidates[start:start+self.batch_size]
            ids, types = pair(np.repeat(query[None], len(uids), axis=0),
                              graph.tokens[[graph.index[u] for u in uids]])
            h = self.ce.joint_features(torch.as_tensor(ids, device=device_of(self.ce)),
                                       torch.as_tensor(types, device=device_of(self.ce)))
            features.append(h.cpu())
            scores.extend(self.ce.classifier(h).squeeze(-1).cpu().tolist())
        return np.asarray(scores), torch.cat(features)

    @torch.inference_mode()
    def rank(self, rows, be_k, ce_k, full=True):
        """CE reranks BE top be_k; output tail retains the remaining BE order."""
        if be_k < 1 or ce_k < 1:
            raise ValueError('Candidate counts must be positive.')
        self.be.eval()
        by_year = defaultdict(list)
        for i, row in enumerate(rows):
            by_year[int(row.get('year', 0))].append(i)
        results = [None]*len(rows)
        for indices in by_year.values():
            graph = self.corpus.graph(rows[indices[0]])
            bank = self.bank(rows[indices[0]]).to(device_of(self.be))
            for start in range(0, len(indices), self.batch_size):
                ix = indices[start:start+self.batch_size]
                tokens = torch.as_tensor(self.corpus.queries([rows[i] for i in ix]), device=device_of(self.be))
                similarity = (self.be.encode(tokens, 'query') @ bank.T).cpu().numpy()
                for j, i in enumerate(ix):
                    row = rows[i]
                    hidden = row.get('target') if row.get('masked', False) else None
                    order = [graph.uids[k] for k in np.argsort(-similarity[j], kind='stable')
                             if graph.uids[k] != hidden]
                    pool = order[:be_k]
                    if self.ce is not None:
                        scores, _ = self.score_pairs(row, pool)
                        pool = [pool[k] for k in sorted(range(len(pool)), key=lambda k: (-scores[k], pool[k]))]
                    ranking = pool + order[be_k:] if full else pool
                    results[i] = {'id': sample_id(row), 'ranking': ranking, 'natural': pool[:ce_k]}
            del bank
        return results



def _configured_inputs(corpus, task, phase, config_path, **explicit):
    if all(v is not None for v in explicit.values()):
        return explicit
    names={'TaxoAP-MeSH':'mesh','TaxoAP-IEEE':'ieee','QEN-MeSH':'qmesh','TaxoAP-IEEE-static':'ieee_static','IEEE-static':'ieee_static'}
    dataset=names.get(corpus.root.name,corpus.root.name)
    values=input_settings(task,dataset,phase,config_path)
    values.update({k:v for k,v in explicit.items() if v is not None})
    return values

def prepare_admission(corpus, rows, retriever=None, be_k=None, ce_k=None, slots=None,
                      max_paths=None, phase="inference", config_path=None):
    """Limit active paths separately from padded tensor slots."""
    values = _configured_inputs(corpus,"admission",phase,config_path,be_k=be_k,ce_k=ce_k,slots=slots,max_paths=max_paths)
    be_k,ce_k,slots,max_paths = (values[k] for k in ("be_k","ce_k","slots","max_paths"))
    if not 1 <= max_paths <= slots:
        raise ValueError('Expected 1 <= max_paths <= slots.')
    results = retriever.rank(rows, be_k, ce_k, full=False) if retriever is not None else None
    examples = []
    for i, row in enumerate(rows):
        example = {'id': sample_id(row), 'query': corpus.queries([row])[0]}
        if 'label' in row:
            example['label'] = row['label']
        if results is not None:
            masked = row.get('target') if row.get('masked', False) else None
            texts, _ = build_memory(results[i]['natural'], corpus.graph(row).records, max_paths, masked)
            example['memory'], example['roles'] = corpus.memory_tokens(texts, slots)
        examples.append(example)
    return examples


@torch.inference_mode()
def predict_admission(model, examples, batch_size=128):
    was_training = model.training
    model.eval()
    out = []
    try:
        for i in range(0, len(examples), batch_size):
            batch = admission_batch(examples[i:i+batch_size], device_of(model))
            batch.pop('labels', None)
            out.extend(model(**batch).cpu().tolist())
    finally:
        model.train(was_training)
    return np.asarray(out)


@torch.inference_mode()
def prepare_placement(corpus, rows, retriever, path_encoder, training=False,
                      be_k=None, top_k=None, slots=None, path_chunk=64, max_paths=None, config_path=None, path_cache=None):
    """Build frozen features and memory; add gold candidates for training."""
    values = _configured_inputs(corpus,"placement","training" if training else "inference",config_path,be_k=be_k,ce_k=top_k,slots=slots,max_paths=max_paths)
    be_k,top_k,slots,max_paths = (values[k] for k in ("be_k","ce_k","slots","max_paths"))
    if training:
        assert_training_rows(rows)
    results = retriever.rank(rows, be_k, top_k, full=not training)
    path_encoder.eval()
    examples = []
    # Deduplicate path encodings within this preparation call.
    path_cache = {} if path_cache is None else path_cache
    for row, result in zip(rows, results):
        natural = result['natural']
        chosen = list(natural)
        gold = set(row.get('gold', []))
        if training:
            if not gold or len(gold) > top_k:
                raise ValueError('Training requires between one and top_k gold parents.')
            if gold-set(chosen):
                chosen = sorted(gold) + [u for u in chosen if u not in gold][:top_k-len(gold)]
        graph = corpus.graph(row)
        masked = row.get('target') if row.get('masked', False) else None
        texts, _ = build_memory(natural, graph.records, max_paths, masked)
        missing = [t for t in texts if t not in path_cache]
        for start in range(0, len(missing), path_chunk):
            part = missing[start:start+path_chunk]
            tokens = corpus.tokenizer(part, max_length=226, truncation=True, padding='max_length', return_tensors='pt')['input_ids']
            vectors = path_encoder(tokens.to(device_of(path_encoder))).cpu()
            path_cache.update(zip(part, vectors))
        memory = torch.zeros(slots, 768)
        memory[:len(texts)] = torch.stack([path_cache[t] for t in texts])
        roles = torch.full((slots,), 3, dtype=torch.long)
        roles[:len(texts)] = 0
        _, features = retriever.score_pairs(row, chosen)
        examples.append({'id': sample_id(row), 'row': row, 'candidates': chosen,
                         'path_texts': texts, 'features': features.clone(), 'memory': memory.clone(), 'roles': roles.clone(),
                         'gold': torch.tensor([u in gold for u in chosen]),
                         'hard2': torch.tensor([u in natural[:2] and u not in gold for u in chosen]),
                         'hard8': torch.tensor([u in natural[:8] and u not in gold for u in chosen]),
                         'tail': [u for u in result['ranking'] if u not in chosen] if not training else []})
    return examples


def placement_batch(examples, device, permute=False, rng=None):
    widths = {len(e['candidates']) for e in examples}
    if len(widths) != 1:
        raise ValueError('Batch examples with equal candidate counts together.')
    batch = {k: torch.stack([e[k] for e in examples]).to(device)
             for k in ('features', 'memory', 'roles', 'gold', 'hard2', 'hard8')}
    if permute:
        rng = rng if rng is not None else np.random.default_rng()
        orders = torch.as_tensor(np.stack([rng.permutation(len(e['candidates'])) for e in examples]), device=device)
        batch['features'] = batch['features'].gather(1, orders[..., None].expand(-1, -1, 768))
        for k in ('gold', 'hard2', 'hard8'):
            batch[k] = batch[k].gather(1, orders)
    return batch


@torch.inference_mode()
def predict_placement(model, examples, batch_size=128):
    was_training = model.training
    model.eval()
    rankings = [None]*len(examples)
    by_width = defaultdict(list)
    for i, e in enumerate(examples):
        by_width[len(e['candidates'])].append(i)
    try:
        for indices in by_width.values():
            for start in range(0, len(indices), batch_size):
                ix = indices[start:start+batch_size]
                batch = placement_batch([examples[i] for i in ix], device_of(model))
                scores = model(batch['features'], batch['memory'], batch['roles']).cpu().numpy()
                for i, values in zip(ix, scores):
                    e = examples[i]
                    order = sorted(range(len(values)), key=lambda k: (-values[k], e['candidates'][k]))
                    rankings[i] = [e['candidates'][k] for k in order]+e['tail']
    finally:
        model.train(was_training)
    return rankings


def end_to_end(query_model, memory_model, admission_examples, placement_model,
               placement_examples, thresholds, batch_size=128):
    """Join stages by sample ID using each task's query text."""
    q = predict_admission(query_model, admission_examples, batch_size)
    m = predict_admission(memory_model, admission_examples, batch_size)
    accepted = (q >= thresholds[0]) & (m >= thresholds[1])
    mapping = {e['id']: e for e in placement_examples}
    if len(mapping) != len(placement_examples):
        raise ValueError('Duplicate placement IDs.')
    selected = [mapping[e['id']] for e, keep in zip(admission_examples, accepted) if keep and e['id'] in mapping]
    ranks = predict_placement(placement_model, selected, batch_size)
    by_id = {e['id']: rank for e, rank in zip(selected, ranks)}
    return [{'id': e['id'], 'accepted': bool(keep), 'query_score': float(qs),
             'memory_score': float(ms), 'ranking': by_id.get(e['id'])}
            for e, keep, qs, ms in zip(admission_examples, accepted, q, m)]
