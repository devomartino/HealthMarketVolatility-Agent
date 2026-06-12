from dotenv import load_dotenv
load_dotenv()

from src.pipeline.database import get_connection, initialize_database, build_joined_view
from src.ml.features import build_feature_matrix

con = get_connection()
initialize_database(con)
build_joined_view(con)

X, y = build_feature_matrix(con)
print(f"Total rows: {len(y)}")
print(f"Stressed days (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Normal days (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")