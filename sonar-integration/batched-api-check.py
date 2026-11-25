"""Exercise continuous batching after the isolated A/B requests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import urllib.request


def check(index):
    rng = random.Random(309000 + index)
    body = {'model': 'qwen3.8-27b', 'prompt': [rng.randrange(1000, 30000) for _ in range(9000)],
            'max_tokens': 64, 'ignore_eos': True, 'temperature': 0, 'seed': 3090,
            'cache_salt': 'batch-verification-' + str(index)}
    request = urllib.request.Request('http://127.0.0.1:2242/v1/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
        assert response.status == 200
    assert result['usage']['prompt_tokens'] == 9000, result['usage']
    assert result['usage']['completion_tokens'] == 64, result['usage']
    assert result['choices'][0]['text']
    return {'request': index, 'elapsed_seconds': time.perf_counter() - started,
            'usage': result['usage'], 'status': 'PASS'}


with ThreadPoolExecutor(max_workers=4) as executor:
    records = list(executor.map(check, range(4)))
output = Path(__file__).parent / 'inference' / 'batched-api-check.json'
assert not output.exists(), 'Preserve previous evidence'
output.write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records), flush=True)
