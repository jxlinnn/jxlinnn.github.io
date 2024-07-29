from langdetect import detect
import pyarrow as pa
from pyarrow import csv

def filter_lang(data: List) -> List, List:
  english = []
  other = []
  for x,y in data:
      try:
          lang = detect(x)
          if lang == 'en':
              review_rating.append((x,y))
      except:
          others.append((x,y))
    return english, other

def convert_to_csv(data: List, columns: List, csv_name: String):
  df = pd.DataFrame(data, columns=columns)
  pa_df = pa.Table.from_pandas(df)
  csv.write_csv(pa_df, csv_name)
