import pandas as pd
import glob
import os

files = glob.glob('backend/data_cache_alpaca/*.parquet')
print(f'Checking {len(files)} files...')

corrupted = []
valid = 0

for f in files:
    try:
        if os.path.getsize(f) < 1000:
            raise ValueError('Too small (<1KB)')
        
        # Try reading header/schema
        pd.read_parquet(f, columns=['close'])
        valid += 1
    except Exception as e:
        corrupted.append((f, str(e)))

print(f'\nSummary:\nValid Files: {valid}\nCorrupted Files: {len(corrupted)}')

if corrupted:
    print("\nCorrupted List:")
    for c in corrupted:
        print(c)
