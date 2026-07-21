import json
import os

transcript_path = r'C:\Users\shrav\.gemini\antigravity\brain\11d9e0cb-6b12-4ec9-ab8f-39dc7dd1cc9c\.system_generated\logs\transcript_full.jsonl'
files_to_recover = [
    'src/data_pipeline.py', 'src/nlp_engine.py', 'src/mbti_engine.py', 
    'src/scoring_engine.py', 'src/feedback_loop.py', 'src/utils.py', 
    'src/__init__.py', 'analysis/performance_report.py', 'app.py'
]

recovered = {}

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'tool_calls' in data:
            for tc in data['tool_calls']:
                if tc['name'] == 'write_to_file':
                    args = tc['args']
                    target = args.get('TargetFile', '').strip()
                    content = args.get('CodeContent', '')
                    for fname in files_to_recover:
                        if os.path.basename(fname) == os.path.basename(target):
                            recovered[fname] = content

print(f'Found {len(recovered)} files to recover.')
for fname, content in recovered.items():
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    with open(fname, 'w', encoding='utf-8') as out:
        out.write(content)
    print(f'Recovered {fname}')
