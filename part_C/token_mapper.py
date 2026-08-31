from transformers import AutoTokenizer
import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_FIELDS = ["timestamp", "user", "location", "component", "content"]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_path = input("window JSONL 파일 경로를 입력하세요:")
    input_path = Path(input_path)

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return

    fin = open(input_path,"r",encoding = "utf-8")

    for line in fin: 
        window = json.loads(line)

        text =""
        spans = []

        for row in window:
          line_id = int(row["line_id"])
          text += f"[LINE {line_id}]"

          for field in INPUT_FIELDS:
              value = row[field]

              text += f"{field}="
              start = len(text)


              text += value
              end = len(text)

              spans.append({
                    "line_id": line_id,
                    "field": field,
                    "start": start,
                    "end": end
                })

              text +=" "

          text += "\n"
        
                          


