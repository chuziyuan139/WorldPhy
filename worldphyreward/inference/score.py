#!/usr/bin/env python3
"""Score caption + conditioning image + video with the fixed WorldPhyReward protocol.

Single-file vLLM inference. Supply the WorldPhyReward checkpoint with --model.
No annotations are included in model inputs. JSON/JSONL media paths are resolved
relative to the records file, unless --media-root is supplied.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile

# Keep CLI inspection available without the GPU inference dependencies.
try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    from vllm.tokenizers.registry import cached_tokenizer_from_config
except ModuleNotFoundError as error:
    if (error.name or '').split('.')[0] != 'vllm':
        raise
    AdapterLogitsProcessor = object
    cached_tokenizer_from_config = None

FPS = 5.0
IMAGE_PIXELS = VIDEO_PIXELS = 640 * 480
MAX_FRAMES = 128
SIZE = (640, 480)
RUBRICS = [f'R{i}' for i in range(1, 8)]
CONS_TOKEN_ID = 14777
ISSUE_TOKEN_ID = 40619
EXTRA_KEY = 'rubric_opening_cons_bias'
DECODER = json.JSONDecoder()



PROTOCOL = json.loads('{"system_prompt": "You evaluate the visual and physical quality of a robot video and its compliance with a task caption. You receive only an initial conditioning image, the candidate video\'s chronological frames sampled at 5 fps, and the caption. The initial image is a reference for the intended initial scene; judge the actual actions and final state from the video, not from what the caption says should happen.\\n\\nEvaluate these seven independent rubrics:\\nR1 - Entity and Identity Consistency: Do the robot, its parts, and all visible objects maintain continuous existence, count, and identity without unexplained appearance, disappearance, duplication, or replacement?\\nR2 - Geometric and Material Plausibility: Do the robot and objects maintain shapes, dimensions, local structures, and deformations that are consistent with their visible materials and mechanical properties?\\nR3 - Connection and Joint Constraint Plausibility: Do connected parts remain attached and move only in ways allowed by their visible joints, hinges, pivots, sliders, or other mechanical constraints?\\nR4 - Object Interpenetration and Clipping: Do the robot, objects, and environment avoid visibly passing through one another or occupying the same solid space?\\nR5 - Contact and Interaction Causality: Are grasping, pushing, pulling, carrying, and other controlled object motions supported by a physically plausible contact or force-transmission path?\\nR6 - Support and Motion Dynamics: Are rest, support, sliding, falling, collisions, trajectory changes, and speed changes consistent with visible support, gravity, contact forces, and inertia?\\nR7 - Caption and Instruction Following: Does the video visibly realize every core object, action, object relation, explicit sequence, and required final state described by the caption?\\n\\nFor each rubric, first give a brief English reason, then its binary score. A score of 1 means this rubric is satisfied; 0 means there is a visible violation or a required task action/final state is missing. Base negative reasons on the specific visible entity, action, or inconsistency. Do not invent hidden events, treat ordinary occlusion as disappearance, or assume a task succeeds merely because it is described. Evaluate each rubric separately; one failure does not automatically fail the others.\\n\\nUse a consistent reason style: begin the reason with \\"Issue: \\" if you find a violation, followed by the observed problem. Otherwise begin it with \\"Consistent: \\", followed by the relevant consistent behavior. Choose this wording from the video evidence; no per-video judgment is supplied to you. This wording is part of the reason, not an extra output field. Keep the complete reason, including this prefix, within 45 English words.\\n\\nReturn one JSON object with exactly the keys R1, R2, R3, R4, R5, R6, R7 in that order. Each value must contain \\"reason\\" followed immediately by \\"score\\", in that order. Each reason must be a concise English sentence of at most 45 words. Each score must be the integer 0 or 1 (not a string), consistent with your reason. No extra keys, markdown, or text outside the JSON. This brief evidence summary is the output reason; do not add a separate reasoning trace.", "user_template": "Initial conditioning image:\\n<image>\\nCandidate video, sampled at 5 frames per second in chronological order:\\n<video>\\nTask caption:\\n{caption}\\n\\nEvaluate this video against R1-R7. Return the required JSON only.", "chat_template": "<|im_start|>system\\nYou evaluate the visual and physical quality of a robot video and its compliance with a task caption. You receive only an initial conditioning image, the candidate video\'s chronological frames sampled at 5 fps, and the caption. The initial image is a reference for the intended initial scene; judge the actual actions and final state from the video, not from what the caption says should happen.\\n\\nEvaluate these seven independent rubrics:\\nR1 - Entity and Identity Consistency: Do the robot, its parts, and all visible objects maintain continuous existence, count, and identity without unexplained appearance, disappearance, duplication, or replacement?\\nR2 - Geometric and Material Plausibility: Do the robot and objects maintain shapes, dimensions, local structures, and deformations that are consistent with their visible materials and mechanical properties?\\nR3 - Connection and Joint Constraint Plausibility: Do connected parts remain attached and move only in ways allowed by their visible joints, hinges, pivots, sliders, or other mechanical constraints?\\nR4 - Object Interpenetration and Clipping: Do the robot, objects, and environment avoid visibly passing through one another or occupying the same solid space?\\nR5 - Contact and Interaction Causality: Are grasping, pushing, pulling, carrying, and other controlled object motions supported by a physically plausible contact or force-transmission path?\\nR6 - Support and Motion Dynamics: Are rest, support, sliding, falling, collisions, trajectory changes, and speed changes consistent with visible support, gravity, contact forces, and inertia?\\nR7 - Caption and Instruction Following: Does the video visibly realize every core object, action, object relation, explicit sequence, and required final state described by the caption?\\n\\nFor each rubric, first give a brief English reason, then its binary score. A score of 1 means this rubric is satisfied; 0 means there is a visible violation or a required task action/final state is missing. Base negative reasons on the specific visible entity, action, or inconsistency. Do not invent hidden events, treat ordinary occlusion as disappearance, or assume a task succeeds merely because it is described. Evaluate each rubric separately; one failure does not automatically fail the others.\\n\\nUse a consistent reason style: begin the reason with \\"Issue: \\" if you find a violation, followed by the observed problem. Otherwise begin it with \\"Consistent: \\", followed by the relevant consistent behavior. Choose this wording from the video evidence; no per-video judgment is supplied to you. This wording is part of the reason, not an extra output field. Keep the complete reason, including this prefix, within 45 English words.\\n\\nReturn one JSON object with exactly the keys R1, R2, R3, R4, R5, R6, R7 in that order. Each value must contain \\"reason\\" followed immediately by \\"score\\", in that order. Each reason must be a concise English sentence of at most 45 words. Each score must be the integer 0 or 1 (not a string), consistent with your reason. No extra keys, markdown, or text outside the JSON. This brief evidence summary is the output reason; do not add a separate reasoning trace.<|im_end|>\\n<|im_start|>user\\nInitial conditioning image:\\n<|vision_start|><|image_pad|><|vision_end|>\\nCandidate video, sampled at 5 frames per second in chronological order:\\n<|vision_start|><|video_pad|><|vision_end|>\\nTask caption:\\n{caption}\\n\\nEvaluate this video against R1-R7. Return the required JSON only.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", "enable_thinking": false, "video_fps": 5.0, "max_frames": 128, "video_pixels": 307200, "image_pixels": 307200, "context_length": 24576, "max_new_tokens": 1024, "temperature": 0.0, "top_p": 1.0, "seed": 20261011, "rubrics": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"], "output_fields": ["reason", "score"], "score_values": [0, 1], "max_reason_words": 45, "attention_backend": "FLASH_ATTN", "flash_attn_version": 2}')
OPENING_BIAS = json.loads('{"schema": "reason_opening_cons_bias_v1", "cons_token_id": 14777, "issue_token_id": 40619, "cons_bias": {"R1": -0.1875, "R2": 0.1875, "R3": -0.0625, "R4": -0.4375, "R5": -0.1875, "R6": -0.1875, "R7": 1.1875}, "scale": 1.0, "shared_for_all_samples": true}')


def load_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.chmod(temporary, 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def target(record, reason_verdict_prefix=False):
    result = {}
    for key, value in record['score'].items():
        reason = record['reason'][key]
        if reason_verdict_prefix:
            reason = ('Consistent: ' if value else 'Issue: ') + reason
            if len(reason.split()) > 45:
                raise ValueError('Reference reason exceeds the fixed 45-word limit')
        result[key.split(' - ', 1)[0]] = {'reason': reason, 'score': int(value)}
    return result

def parse_prediction(text):
    raw = text.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
    data = json.loads(raw)
    if not isinstance(data, dict) or list(data) != RUBRICS:
        raise ValueError('Output must contain R1-R7 in order')
    for r in RUBRICS:
        if not isinstance(data[r], dict) or list(data[r]) != ['reason', 'score']:
            raise ValueError(f'{r}: expected reason then score')
        if type(data[r]['score']) is not int or data[r]['score'] not in (0, 1):
            raise ValueError(f'{r}: score is not integer 0/1')
        if not isinstance(data[r]['reason'], str) or not data[r]['reason'].strip():
            raise ValueError(f'{r}: reason missing')
        if len(data[r]['reason'].split()) > 45:
            raise ValueError(f'{r}: reason exceeds 45 words')
    return data

def parse_result(result):
    """Reject protocol violations even when the preserved raw JSON is valid."""
    if result.get('protocol_error') or result.get('tool_calls'):
        raise ValueError(result.get('protocol_error') or 'Unexpected tool use in label-blind baseline')
    return parse_prediction(result['text'])

def class_metrics(truth, predicted, positive):
    tp = sum(t == positive and p == positive for t, p in zip(truth, predicted))
    fp = sum(t != positive and p == positive for t, p in zip(truth, predicted))
    fn = sum(t == positive and p != positive for t, p in zip(truth, predicted))
    # An invalid prediction is a miss for its ground-truth class, not a correct negative.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'support': sum(t == positive for t in truth), 'tp': tp, 'fp': fp, 'fn': fn}

def evaluate(records, predictions):
    gold = {r['video']: target(r) for r in records}
    by_video = {p['video']: p for p in predictions}
    if len(by_video) != len(predictions) or not set(by_video) <= set(gold):
        raise ValueError('Duplicate or unknown prediction records')
    per = {}
    all_t, all_p, exact, valid, any_t, any_p = [], [], 0, 0, [], []
    parsed = {}
    for video in gold:
        try:
            parsed[video] = parse_result(by_video[video])
            valid += 1
        except (KeyError, ValueError, TypeError):
            parsed[video] = None
    for rubric in RUBRICS:
        truth = [gold[v][rubric]['score'] for v in gold]
        pred = [parsed[v][rubric]['score'] if parsed[v] else None for v in gold]
        per[rubric] = {'fail_0': class_metrics(truth, pred, 0), 'pass_1': class_metrics(truth, pred, 1),
                       'accuracy': sum(t == p for t, p in zip(truth, pred)) / len(truth)}
        all_t.extend(truth)
        all_p.extend(pred)
    for video, labels in gold.items():
        p = parsed[video]
        exact += int(p is not None and all(p[r]['score'] == labels[r]['score'] for r in RUBRICS))
        any_t.append(int(any(labels[r]['score'] == 0 for r in RUBRICS)))
        any_p.append(int(any(p[r]['score'] == 0 for r in RUBRICS)) if p else None)
    macro = {c: {m: sum(per[r][c][m] for r in RUBRICS) / len(RUBRICS)
                 for m in ['precision', 'recall', 'f1']} for c in ['fail_0', 'pass_1']}
    minimum_pr = min(per[r][c][m] for r in RUBRICS
                     for c in ['fail_0', 'pass_1'] for m in ['precision', 'recall'])
    return {'total_videos': len(records), 'generated_videos': len(by_video),
            'valid_outputs': valid, 'invalid_or_missing_outputs': len(records) - valid,
            'per_rubric': per, 'overall_micro': {'fail_0': class_metrics(all_t, all_p, 0),
                                                'pass_1': class_metrics(all_t, all_p, 1)},
            'overall_macro': macro,
            'rubric_accuracy': sum(t == p for t, p in zip(all_t, all_p)) / len(all_t),
            'all_seven_exact_match': exact / len(records),
            'video_any_failure': class_metrics(any_t, any_p, 1),
            'minimum_per_rubric_precision_recall': minimum_pr,
            'reason_note': 'Reasons are generated, stored and format-checked; these metrics assess scores, not semantic reason correctness.'}

def resize_image(image):
    from PIL import Image
    image = image.convert('RGB')
    if image.size != SIZE:
        image = image.resize(SIZE, resample=Image.Resampling.BICUBIC)
    return image

def resized(image, max_pixels, min_pixels=None):
    if max_pixels != 640 * 480:
        raise ValueError('This isolated run requires a 640x480 pixel budget')
    return resize_image(image)

def load_media(image_path, video_path, fps=FPS, max_frames=MAX_FRAMES,
               video_pixels=VIDEO_PIXELS, image_pixels=IMAGE_PIXELS,
               image_min_pixels=None, video_min_pixels=None):
    import av
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        image = resized(image, image_pixels, image_min_pixels)
    with av.open(str(video_path), 'r') as container:
        stream = container.streams.video[0]
        if stream.duration is None or not stream.frames or stream.average_rate is None:
            raise ValueError(f'Missing temporal metadata: {video_path}')
        duration = float(stream.duration * stream.time_base)
        num_samples = max(1, math.floor(duration * fps))
        if num_samples > max_frames:
            raise ValueError(f'{fps:g} fps exceeds the frame budget; do not silently subsample')
        n = min(stream.frames, num_samples)
        indices = np.linspace(0, stream.frames - 1, n).astype(np.int32).tolist()
        wanted = set(indices)
        frames = [np.asarray(resized(frame.to_image(), video_pixels, video_min_pixels))
                  for i, frame in enumerate(container.decode(stream)) if i in wanted]
        times_as_indices = [i / float(stream.average_rate) * fps for i in indices]
    if len(frames) != n:
        raise ValueError(f'Expected {n} sampled frames, decoded {len(frames)}')
    if len(frames) % 2:
        frames.append(frames[-1])
    metadata = {'fps': fps, 'duration': duration, 'total_num_frames': len(frames),
                'frames_indices': times_as_indices, 'video_backend': 'pyav', 'do_sample_frames': False}
    return image, np.stack(frames), metadata

def validate_bias(value):
    if not isinstance(value, dict) or set(value) != set(RUBRICS):
        raise ValueError('Opening bias must contain exactly R1-R7')
    if any(type(value[r]) not in (int, float) or not math.isfinite(value[r])
           or abs(value[r]) > 20 for r in RUBRICS):
        raise ValueError('Opening biases must be finite numbers with abs <= 20')
    return tuple(float(value[r]) for r in RUBRICS)

def reason_opening_rubric(text):
    """Identify only an empty reason string in the expected JSON structure.

    Previous strings are parsed by the standard JSON decoder, so escaped quotes
    or fake rubric names inside a reason cannot trigger another rubric's bias.
    Partial/malformed or different structures receive no bias and remain subject
    to the unchanged final strict parser.
    """
    pos, end = 0, len(text)

    def space():
        nonlocal pos
        while pos < end and text[pos] in ' \t\r\n':
            pos += 1

    def literal(value):
        nonlocal pos
        space()
        if not text.startswith(value, pos):
            raise ValueError('Incomplete or different JSON structure')
        pos += len(value)

    def value():
        nonlocal pos
        space()
        item, pos = DECODER.raw_decode(text, pos)
        return item

    try:
        space()
        if text.startswith('```', pos):
            match = re.match(r'```(?:json)?\s*', text[pos:])
            pos += match.end()
        literal('{')
        for rubric in RUBRICS:
            if value() != rubric:
                return None
            literal(':')
            literal('{')
            if value() != 'reason':
                return None
            literal(':')
            space()
            if pos >= end or text[pos] != '"':
                return None
            # Treat whitespace inside an otherwise empty reason as preceding its
            # first content token. No ordinary nonempty reason gets modified.
            if not text[pos + 1:].strip(' \t\r\n'):
                return rubric
            if not isinstance(value(), str):
                return None
            literal(',')
            if value() != 'score':
                return None
            literal(':')
            score = value()
            if type(score) is not int or score not in (0, 1):
                return None
            literal('}')
            if rubric != RUBRICS[-1]:
                literal(',')
    except (ValueError, TypeError, IndexError):
        return None
    return None

class RequestOpeningBias:
    def __init__(self, tokenizer, biases, trace=False):
        self.tokenizer = tokenizer
        self.biases = tuple(biases)
        self.enabled = any(self.biases)
        self.trace = trace
        self.traced = set()

    def __call__(self, output_ids, logits):
        # Exact no-op, including object identity and floating-point bits.
        if not self.enabled:
            return logits
        # Explicit LLM registration makes vLLM repair async placeholders before
        # AdapterLogitsProcessor.apply calls this function. Never decode -1.
        if any(token < 0 for token in output_ids):
            raise RuntimeError('Async output token placeholder reached opening processor')
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        rubric = reason_opening_rubric(text)
        if rubric is None:
            return logits
        bias = self.biases[RUBRICS.index(rubric)]
        if bias:
            logits[CONS_TOKEN_ID] += bias
            if self.trace and rubric not in self.traced:
                print(json.dumps({'event': 'reason_opening_bias_applied', 'rubric': rubric,
                                  'output_prefix_tokens': len(output_ids),
                                  'modified_token_id': CONS_TOKEN_ID, 'bias': bias,
                                  'other_logits_unchanged': True}), flush=True)
                self.traced.add(rubric)
        return logits

class RubricOpeningBiasProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        extra = sampling_params.extra_args or {}
        if set(extra) != {EXTRA_KEY}:
            raise ValueError('This processor accepts only the global opening bias vector')
        validate_bias(extra[EXTRA_KEY])

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if self.tokenizer is None:
            raise ValueError('Opening processor requires the model tokenizer')
        if self.tokenizer.decode([CONS_TOKEN_ID]) != 'Cons' or self.tokenizer.decode([ISSUE_TOKEN_ID]) != 'Issue':
            raise ValueError('Qwen reason-opening token ids differ from the fixed contract')
        self.requests_created = 0
        self.global_biases = None

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        self.validate_params(params)
        biases = validate_bias(params.extra_args[EXTRA_KEY])
        if self.global_biases is None:
            self.global_biases = biases
        elif biases != self.global_biases:
            raise ValueError('Every request in this engine must share one seven-value bias')
        self.requests_created += 1
        return RequestOpeningBias(self.tokenizer, biases, trace=self.requests_created == 1)

def observe_openings(result, tokenizer):
    ids = list(result.token_ids)
    rows = result.logprobs
    observed = {rubric: {'Issue': None, 'Consistent': None,
                         'status': 'reason_opening_not_found'} for rubric in RUBRICS}
    if rows is None or len(rows) != len(ids):
        return {key: dict(value, status='token_logprobs_missing')
                for key, value in observed.items()}
    # Decode actual generated token prefixes, rather than retokenizing text
    # and silently assigning its potentially different boundaries.
    prefixes = [''] + [tokenizer.decode(ids[:end], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
                       for end in range(1, len(ids) + 1)]
    if prefixes[-1] != result.text:
        return {key: dict(value, status='emitted_token_text_alignment_unavailable')
                for key, value in observed.items()}
    pattern = r'"(R[1-7])"\s*:\s*\{\s*"reason"\s*:\s*"\s*(Issue|Consistent)\b'
    for match in re.finditer(pattern, result.text):
        rubric, word = match.group(1), match.group(2)
        opening_start = match.start(2)
        index = next((i for i in range(len(ids))
                      if result.text.startswith(prefixes[i])
                      and result.text.startswith(prefixes[i + 1])
                      and len(prefixes[i]) <= opening_start < len(prefixes[i + 1])), None)
        if index is None:
            observed[rubric]['status'] = 'opening_token_boundary_unavailable'
            continue
        item = {'Issue': None, 'Consistent': None, 'status': 'observed',
                'generated_opening': word, 'generated_token_index': index,
                'generated_token_id': ids[index],
                'generated_token_text': prefixes[index + 1][len(prefixes[index]):],
                'token_start_character': len(prefixes[index]),
                'opening_start_character': opening_start,
                'candidates': {}}
        distribution = rows[index] or {}
        for alternative in ('Issue', 'Consistent'):
            # The alternative's tokenization must preserve the exact emitted
            # history up to this boundary. Otherwise comparison is undefined.
            probe = tokenizer.encode(result.text[:opening_start] + alternative,
                                     add_special_tokens=False)
            if len(probe) <= index or probe[:index] != ids[:index]:
                item['candidates'][alternative] = {'token_id': None, 'logprob': None,
                                                  'status': 'tokenization_not_aligned'}
                continue
            candidate = probe[index]
            decoded = tokenizer.decode(probe[:index + 1], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            if len(decoded) <= opening_start:
                item['candidates'][alternative] = {'token_id': None, 'logprob': None,
                                                  'status': 'first_word_not_in_this_token'}
                continue
            value = distribution.get(candidate)
            logprob = float(value.logprob) if value is not None else None
            item[alternative] = logprob
            item['candidates'][alternative] = {
                'token_id': candidate, 'token_text': decoded[len(prefixes[index]):],
                'logprob': logprob,
                'status': 'present' if value is not None else 'not_in_returned_top20'}
        observed[rubric] = item
    return observed

def inputs(args, parser):
    if args.records:
        if any(x is not None for x in [args.video, args.image, args.caption]):
            parser.error('Use either --records or --video/--image/--caption')
        path = args.records.resolve()
        rows = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if path.suffix.lower() == '.jsonl' else load_json(path))
        if isinstance(rows, dict):
            rows = rows.get('records', rows.get('videos'))
        root = (args.media_root or path.parent).resolve()
    else:
        if any(x is None for x in [args.video, args.image, args.caption]):
            parser.error('Provide --records, or all of --video, --image and --caption')
        rows = [{'video': str(args.video.resolve()), 'image': str(args.image.resolve()), 'caption': args.caption}]
        root = (args.media_root or Path.cwd()).resolve()
    if not isinstance(rows, list) or not rows:
        parser.error('Records must be a nonempty JSON list')
    if len({r['video'] for r in rows}) != len(rows):
        parser.error('Duplicate video identifiers')
    for row in rows:
        if not isinstance(row.get('caption'), str) or not row['caption'].strip():
            parser.error('Every record needs a nonempty caption')
        for key in ['image', 'video']:
            if not (root / row[key]).is_file():
                parser.error(f'Missing {key}: {root / row[key]}')
    return rows, root

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='WorldPhyReward checkpoint directory or Hugging Face model ID')
    p.add_argument('--records', type=Path)
    p.add_argument('--media-root', type=Path, help='Relative paths default to the records file directory')
    p.add_argument('--video', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--caption')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.tp < 1 or a.batch_size < 1:
        p.error('--tp and --batch-size must be positive')
    rows, root = inputs(a, p)
    settings = dict(PROTOCOL)
    settings['seed'] = a.seed
    bias = OPENING_BIAS
    model = os.path.expanduser(a.model)
    context = {'model': str(model), 'records': str(a.records.resolve()) if a.records else None,
               'media_root': str(root), 'evaluation': settings, 'decoding_adjustment': bias,
               'tensor_parallel_size': a.tp, 'batch_size': a.batch_size}
    if a.dry_run:
        print(__import__('json').dumps(dict(context, videos=len(rows)), ensure_ascii=False, indent=2))
        return
    os.umask(0)
    a.output.mkdir(parents=True, exist_ok=True)
    a.output.chmod(0o777)
    with (a.output / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a, rows, root, model, settings, bias, context)

def run(a, rows, root, model, settings, bias, context):
    context_path = a.output / 'context.json'
    prediction_path = a.output / 'predictions.json'
    if context_path.exists() and load_json(context_path) != context:
        raise ValueError('Output directory already belongs to a different input/model/protocol')
    if prediction_path.exists() and not context_path.exists():
        raise ValueError('Existing predictions lack their protocol context')
    save_json(context_path, context)
    existing = load_json(prediction_path) if prediction_path.exists() else []
    done = {r['video']: r for r in existing}
    if len(done) != len(existing) or not set(done) <= {r['video'] for r in rows}:
        raise ValueError('Duplicate or unrelated existing predictions')
    pending = [r for r in rows if r['video'] not in done]
    if pending:
        os.environ.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        if a.tp != torch.cuda.device_count():
            raise ValueError('--tp must equal the number of visible allocated GPUs')
        save_json(a.output / 'runtime.json', {'torch': torch.__version__, 'vllm': vllm.__version__,
            'cuda': torch.version.cuda, 'gpu_names': [torch.cuda.get_device_name(i) for i in range(a.tp)]})
        llm = LLM(model=str(model), tensor_parallel_size=a.tp, dtype='bfloat16',
            trust_remote_code=False, max_model_len=settings['context_length'],
            max_num_batched_tokens=settings['context_length'], max_num_seqs=a.batch_size,
            gpu_memory_utilization=.85, enforce_eager=True, enable_prefix_caching=False,
            disable_custom_all_reduce=True, gdn_prefill_backend='triton',
            attention_config={'backend': settings['attention_backend'], 'flash_attn_version': settings['flash_attn_version']},
            limit_mm_per_prompt={'image': {'count': 1, 'width': 640, 'height': 480},
                                 'video': {'count': 1, 'num_frames': settings['max_frames'], 'width': 640, 'height': 480}},
            mm_processor_cache_gb=0, seed=a.seed, logits_processors=[RubricOpeningBiasProcessor])
        params = SamplingParams(temperature=settings['temperature'], top_p=settings['top_p'],
            max_tokens=settings['max_new_tokens'], seed=a.seed, stop=['<|im_end|>'], logprobs=20,
            extra_args={EXTRA_KEY: bias['cons_bias']})
        tokenizer = llm.get_tokenizer()
        for start in range(0, len(pending), a.batch_size):
            batch = pending[start:start + a.batch_size]
            requests, media_info = [], []
            for row in batch:
                image, frames, metadata = load_media(root / row['image'], root / row['video'],
                    fps=settings['video_fps'], max_frames=settings['max_frames'],
                    video_pixels=settings['video_pixels'], image_pixels=settings['image_pixels'])
                media_info.append({'video_fps': settings['video_fps'], 'sampled_video_frames': len(frames),
                    'video_duration_seconds': metadata['duration'], 'image_size': list(image.size),
                    'video_frame_size': [int(frames.shape[2]), int(frames.shape[1])]})
                # Only caption, image and frames enter the model. Labels/reasons are never input.
                requests.append({'prompt': settings['chat_template'].replace('{caption}', row['caption']),
                    'multi_modal_data': {'image': image, 'video': [(frames, metadata)]},
                    'mm_processor_kwargs': {'do_sample_frames': False, 'fps': settings['video_fps']}})
            outputs = llm.generate(requests, params, use_tqdm=False)
            for row, output, media in zip(batch, outputs, media_info):
                result = output.outputs[0]
                pred = {'video': row['video'], 'text': result.text, 'finish_reason': result.finish_reason,
                    'prompt_tokens': len(output.prompt_token_ids), 'output_tokens': len(result.token_ids),
                    'reason_opening_logprobs': observe_openings(result, tokenizer), **media}
                try:
                    pred['prediction'] = parse_prediction(result.text)
                    pred['format_error'] = None
                except (ValueError, TypeError) as error:
                    pred['prediction'] = None
                    pred['format_error'] = str(error)
                done[row['video']] = pred
            save_json(prediction_path, [done[r['video']] for r in rows if r['video'] in done])
            save_json(a.output / 'progress.json', {'completed': len(done), 'total': len(rows)})
            print(f'Scored {len(done)}/{len(rows)}', flush=True)
    ordered = [done[r['video']] for r in rows]
    if all('score' in row and 'reason' in row for row in rows):
        save_json(a.output / 'metrics.json', evaluate(rows, ordered))
    save_json(a.output / 'progress.json', {'state': 'complete', 'completed': len(done), 'total': len(rows)})

if __name__ == '__main__':
    main()
