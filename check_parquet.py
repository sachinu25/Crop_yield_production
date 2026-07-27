import pandas as pd
df = pd.read_parquet('models/inference_lookup.parquet')
print([c for c in df.columns if 'precip' in c or 'temp' in c])
