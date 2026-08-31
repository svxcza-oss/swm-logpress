import csv
from pathlib import path 


# ============================================================
# 설정
# ============================================================

LINE_ID_START = 0
PROGRESS_EVERY = 1_000_000


# ============================================================
# CSV에 저장할 필드
# ============================================================

FIELDS = [
    "line_id",
    "label",
    "timestamp",
    "date",
    "user",
    "month",
    "day",
    "time",
    "location",
    "component",
    "content",
]


# ============================================================
# Thunderbird 로그 한 줄 파싱
# ============================================================

def parse_line(line):

    # 줄 끝의 엔터 제거
    line = line.rstrip("\n")

    # 공백 기준 최대 10개로 분리
    p = line.split(None, 9)

    # 정상적으로 파싱할 수 없는 경우
    if len(p) < 9:

        row = {field: "" for field in FIELDS}

        # 원본 로그는 버리지 않고 content에 저장
        row["content"] = line

        return row, False


    # 정상적으로 파싱된 경우
    row = {

        "line_id": None,

        "label": p[0],

        "timestamp": p[1],

        "date": p[2],

        "user": p[3],

        "month": p[4],

        "day": p[5],

        "time": p[6],

        "location": p[7],

        "component": p[8].rstrip(":"),

        "content": p[9] if len(p) > 9 else "",
    }

    return row, True


# ============================================================
# 프로그램 실행
# ============================================================

def main():

    # --------------------------------------------------------
    # 사용자가 직접 로그 파일 경로 입력
    # --------------------------------------------------------

    input_path = input(
        "Thunderbird 로그 파일 경로를 입력하세요: "
    )

    # 혹시 "경로" 형태로 입력했을 경우 따옴표 제거
    input_path = input_path.strip().strip('"')

    INPUT = Path(input_path)


    # --------------------------------------------------------
    # 입력 파일 존재 여부 확인
    # --------------------------------------------------------

    if not INPUT.exists():

        print()
        print("파일을 찾을 수 없습니다.")
        print("입력한 경로:")
        print(INPUT)

        return


    # --------------------------------------------------------
    # 출력 파일 이름 입력
    # --------------------------------------------------------

    output_name = input(
        "저장할 CSV 파일 이름을 입력하세요 "
        "(엔터 = Thunderbird_parsed.csv): "
    )

    output_name = output_name.strip()


    # 아무것도 입력하지 않았으면 기본 이름 사용
    if output_name == "":
        output_name = "Thunderbird_parsed.csv"


    OUTPUT = Path(output_name)


    # --------------------------------------------------------
    # 확인
    # --------------------------------------------------------

    print()
    print("==============================")
    print("파싱 시작")
    print("==============================")

    print("입력 파일:")
    print(INPUT)

    print()

    print("출력 파일:")
    print(OUTPUT)

    print()


    # --------------------------------------------------------
    # 카운트
    # --------------------------------------------------------

    total_count = 0
    fail_count = 0


    # --------------------------------------------------------
    # 파일 열기
    # --------------------------------------------------------

    with open(
        INPUT,
        "r",
        encoding="utf-8-sig",
        errors="ignore"
    ) as fin, open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as fout:


        writer = csv.DictWriter(
            fout,
            fieldnames=FIELDS
        )


        # CSV 첫 번째 줄에 필드 이름 저장
        writer.writeheader()


        # ----------------------------------------------------
        # Thunderbird 로그 한 줄씩 읽기
        # ----------------------------------------------------

        for i, line in enumerate(fin):

            row, parse_ok = parse_line(line)


            # line 번호 저장
            row["line_id"] = i + LINE_ID_START


            # CSV에 한 줄 저장
            writer.writerow(row)


            total_count += 1


            # 파싱 실패
            if not parse_ok:
                fail_count += 1


            # 진행 상황 출력
            if total_count % PROGRESS_EVERY == 0:

                print(
                    f"{total_count:,} 줄 처리...",
                    flush=True
                )


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    print()
    print("==============================")
    print("Thunderbird 파싱 완료")
    print("==============================")

    print(f"전체 로그 수 : {total_count:,}")

    print(
        f"파싱 성공    : "
        f"{total_count - fail_count:,}"
    )

    print(
        f"파싱 실패    : "
        f"{fail_count:,}"
    )


    if total_count > 0:

        fail_ratio = (
            fail_count
            / total_count
            * 100
        )

        print(
            f"실패 비율    : "
            f"{fail_ratio:.6f}%"
        )


    print()
    print("저장 위치:")
    print(OUTPUT.resolve())


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()
