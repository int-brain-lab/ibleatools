from pathlib import Path

path = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics/output/73860622-fdff-46f9-a9e7-bff82ca19dec")

from ephysatlas.utils import get_aggregated_snippets_df

df = get_aggregated_snippets_df(path)

print(df)