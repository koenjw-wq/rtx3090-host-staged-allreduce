import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

label = sys.argv[1]
work = Path(__file__).resolve().parent / 'inference'
output = work / (label + '.jsonl')
if output.exists():
    raise RuntimeError('Preserve the existing run and select a new label')
base = 'http://127.0.0.1:2242'
for attempt in range(180):
    try:
        with urllib.request.urlopen(base + '/health', timeout=3) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(5)
else:
    raise RuntimeError('Qwen health check did not succeed')
with urllib.request.urlopen(base + '/v1/models') as response:
    models = json.load(response)
assert any(model['id'] == 'qwen3.8-27b' for model in models['data'])
print(label, 'service ready', flush=True)

for warmup, iteration in [(True, 0), (False, 1), (False, 2), (False, 3)]:
    for size in [5000, 9000]:
        rng = random.Random(3090 + size + 101 * iteration)
        prompt = [rng.randrange(1000, 30000) for _ in range(size)]
        body = {'model': 'qwen3.8-27b', 'prompt': prompt, 'max_tokens': 16,
                'temperature': 0, 'seed': 3090, 'ignore_eos': True,
                'cache_salt': 'host-staged-' + str(iteration) + '-' + str(size),
                'stream': True, 'stream_options': {'include_usage': True}}
        request = urllib.request.Request(base + '/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        first = None
        text = ''
        usage = None
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                item = json.loads(payload)
                for choice in item.get('choices', []):
                    content = choice.get('text', '')
                    if content:
                        if first is None:
                            first = time.perf_counter() - started
                        text += content
                if item.get('usage'):
                    usage = item['usage']
        elapsed = time.perf_counter() - started
        assert first is not None and usage and usage['prompt_tokens'] == size, (first, usage)
        assert usage['completion_tokens'] == 16, usage
        result = {'label': label, 'warmup': warmup, 'iteration': iteration,
                  'prompt_tokens': size, 'ttft_seconds': first, 'elapsed_seconds': elapsed,
                  'completion_text': text, 'usage': usage}
        with output.open('a') as file:
            file.write(json.dumps(result) + '\n')
        print(json.dumps({key: value for key, value in result.items() if key not in ['completion_text', 'usage']}), flush=True)
print('INFERENCE_BENCHMARK_COMPLETE', flush=True)
