import pandas as pd

# Function to convert pd.Timestamp and pd.NaT to a serializable format
# Custom JSON Encoder class
from json import JSONEncoder


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() if not pd.isnull(obj) else None
        elif pd.isna(obj):
            return None
        return JSONEncoder.default(self, obj)
