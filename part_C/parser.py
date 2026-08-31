import csv 
from pathlib import Path

fields = ["line_id", "label", "timestamp", "date", "user", "month", "day", "time", "location", "component", "content"]
        
def parse_line(line):
    parts = line.rstrip("\n").split(None, 9)

    if len(parts) < 9: 
        row = {field: "" for field in fields}
        row["content"] = line.rstrip("\n")
        return row

    return {
        "line_id": None,
        "label": parts[0],
        "timestamp": parts[1],
        "date": parts[2],
        "user": parts[3],
        "month": parts[4],
        "day": parts[5],
        "time": parts[6],
        "location": parts[7],
        "component": parts[8].rstrip(":"),
        "content": parts[9] if len(parts) > 9 else "",
    }

def main():
    input_path = input("입력 로그 파일 경로를 입력하세요")
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_parsed.csv")

    if not input_path.exists():
        print("입력 파일을 찾을 수 없습니다.")
        return 
    fin = open(input_path,"r",encoding="utf-8-sig", errors = "ignore")
    fout = open(output_path, "w" , newline ="",encoding ="utf-8")
    writer = csv.DictWriter(fout,fieldnames = fields)
    writer.writeheader()

    for i,line in enumerate(fin): 
        row = parse_line(line)
        row["line_id"] = i
        writer.writerow(row)

    fin.close()
    fout.close()
    print(f"완료: {output_path}")

if __name__ == "__main__":
    main()   

    

