"""Round-robin ancestor-path memory from naturally retrieved candidate UIDs."""
def build_memory(candidates, records, max_paths=64, masked_target=None):
    if max_paths < 1 or not candidates:
        raise ValueError('A nonempty candidate list and positive path limit are required.')
    options = []
    for uid in candidates:
        paths = sorted(set(tuple(p + [uid]) for p in records[uid]['paths'])) or [(uid,)]
        for path in paths:
            if masked_target is not None and masked_target in path:
                raise ValueError('Masked target leaked into memory.')
            if not all(u in records for u in path):
                raise ValueError('Unknown path node.')
        options.append([' [SEP] '.join(records[u]['name'] for u in path) for path in paths])
    chosen, seen = [], set()
    for level in range(max(map(len, options))):
        for paths in options:
            if level < len(paths) and paths[level] not in seen:
                seen.add(paths[level]); chosen.append(paths[level])
                if len(chosen) == max_paths:
                    return chosen, [0] * len(chosen)
    return chosen, [0] * len(chosen)

# Valid path role is 0; pad the encoded memory with zeros and its roles with 3.
