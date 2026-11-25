"""Send independent prompts concurrently through Sonar continuous batching."""

from concurrent.futures import ThreadPoolExecutor
import json

from api_bench import sample

sizes = [1024, 2048, 3072, 5120]
with ThreadPoolExecutor(max_workers=len(sizes)) as executor:
    responses = list(executor.map(lambda n: sample(n, 23, "v2-batch", 32), sizes))
assert len(responses) == len(sizes)
print(json.dumps(responses, indent=2))
