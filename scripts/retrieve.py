#!/usr/bin/env python3
"""Small BGE-M3 retrieval demo; returns evidence without loading an answer LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/default.yaml')
    parser.add_argument('--embedding-model', help='Local BGE-M3 directory or model ID')
    parser.add_argument('--reader-model', help='Qwen3-8B tokenizer directory or model ID')
    parser.add_argument('--language', choices=('en', 'zh'), default='en')
    parser.add_argument('--allocation', choices=('residual', 'uniform'))
    parser.add_argument('--device', default='cpu', help='cpu or visible cuda:0; set CUDA_VISIBLE_DEVICES externally')
    args = parser.parse_args(argv)
    import numpy as np
    import yaml
    from transformers import AutoTokenizer
    from FlagEmbedding import BGEM3FlagModel
    from nltk.tokenize import PunktTokenizer
    from facet.pipeline import build_index, retrieve

    config = yaml.safe_load(args.config.read_text())
    if (config['chunking']['complete_sentences_per_group'] != 3
            or config['retrieval']['vector_budget'] != 'B+min(B,S)'
            or config['retrieval']['context_prefix_tokens'] != 64
            or config['retrieval']['title_prefix_tokens'] != 16
            or not config['retrieval']['contextualize']
            or not config['models']['embedding_lowercase']
            or not config['models']['embedding_normalize']):
        raise ValueError('This demo requires three-sentence/B+min(B,S)/64+16-prefix/X representation; context thresholds cannot score raw entries')
    bge_path = args.embedding_model or config['models']['embedding']
    qwen_path = args.reader_model or config['models']['reader']
    bge = AutoTokenizer.from_pretrained(bge_path)
    qwen = AutoTokenizer.from_pretrained(qwen_path)
    if args.device != 'cpu' and not args.device.startswith('cuda:'):
        parser.error('--device must be cpu or cuda:N in the visible device mask')
    model = BGEM3FlagModel(bge_path, use_fp16=args.device != 'cpu', devices=[args.device])

    def encode(texts):
        lengths = [len(bge.encode(t, add_special_tokens=True, truncation=False)) for t in texts]
        if not texts or max(lengths) > 8192:
            raise ValueError('BGE input exceeds 8192; silent truncation is disabled')
        raw = np.asarray(model.encode(texts, batch_size=16, max_length=8192)['dense_vecs'], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 1024:
            raise ValueError('Expected BGE-M3 1024D dense vectors')
        return raw

    calibration = Path(config['retrieval']['calibration_file'])
    if not calibration.is_absolute():
        calibration = args.config.parent.parent / calibration
    thresholds = json.loads(calibration.read_text())['languages'][args.language]
    index = build_index(args.document.read_bytes().decode('utf-8'), encode, qwen, bge,
        PunktTokenizer('english'), allocation=args.allocation or config['retrieval']['allocation'],
        contextualize=config['retrieval']['contextualize'],
        target_tokens=config['chunking']['target_qwen_tokens'])
    result = retrieve(index, args.query, encode, qwen, thresholds=thresholds,
        budget=config['retrieval']['budget_tokens'], order=config['retrieval']['order'])
    print(json.dumps({'retrieval': result, 'index_summary': {'blocks': len(index['blocks']),
        'vectors': len(index['entries']), 'metadata': index['metadata']}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
