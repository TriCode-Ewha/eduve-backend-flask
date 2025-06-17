# test_search_accuracy.py  ✱ 수정: Top‑5 페이지 일치 인정
"""Search API 정밀 평가 (Top‑1 파일 / Top‑5 페이지)
--------------------------------------------------
• tests.json / csv  →  query, answer_filename, answer_page
• /search POST → 결과 5개 받음
  - 파일 정확도 : Top‑1 file_name 일치 여부
  - 페이지 정확도 : Top‑5 안에 (file_name & page) 모두 일치 항목 존재
• 결과 CSV 2종 저장 (all_cases / wrong_cases)
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import requests, pandas as pd
from tqdm import tqdm

# ───────────────────────── 데이터 로드 ─────────────────────────

def load_tests(path: Path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ───────────────────────── 평가 함수 ─────────────────────────

def evaluate(api_url: str, tests: list[dict], user_id: str, teacher_id: str | None = None):
    cnt_file_ok = cnt_page_ok = 0
    wrong_cases, all_rows = [], []

    for t in tqdm(tests, desc="Evaluating", unit="q"):
        q, gt_file = t["query"], t["answer_filename"]
        gt_page = int(t["answer_page"]) if str(t.get("answer_page")).isdigit() else None

        payload = {"query": q, "userId": user_id}
        if teacher_id:
            payload["teacherId"] = teacher_id

        try:
            r = requests.post(api_url, json=payload, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print("❌", e, "→", q)
            continue

        hits = r.json().get("results", [])
        if not hits:
            wrong_cases.append({"query": q, "reason": "no_result", "gt": (gt_file, gt_page)})
            continue

        # A) 파일 정확도 : Top‑1 비교
        file_ok = hits[0]["file_name"] == gt_file
        if file_ok:
            cnt_file_ok += 1

        # B) 페이지 정확도 : Top‑5 중 하나라도 (file+page) 일치
        if gt_page is None:
            page_ok = file_ok  # page 무시 항목은 파일만 맞으면 OK
        else:
            page_ok = any(h["file_name"] == gt_file and h.get("page") == gt_page for h in hits)
        if page_ok:
            cnt_page_ok += 1
        else:
            wrong_cases.append({
                "query": q,
                "gt_file": gt_file,
                "gt_page": gt_page,
                "top5": [(h["file_name"], h.get("page"), h.get("score")) for h in hits]
            })

        all_rows.append({
            **t,
            "pred_file_top1": hits[0]["file_name"],
            "pred_page_top1": hits[0].get("page"),
            "score_top1":     hits[0].get("score"),
            "correct_file":   file_ok,
            "correct_page":   page_ok
        })

    total = len(tests) or 1
    return cnt_file_ok/total*100, cnt_page_ok/total*100, wrong_cases, all_rows

# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--host", default="http://localhost:5000")
    ap.add_argument("--user", required=True)
    ap.add_argument("--teacher")
    args = ap.parse_args()

    tests = load_tests(Path(args.data))
    file_acc, page_acc, wrong, all_rows = evaluate(f"{args.host.rstrip('/')}/search", tests, args.user, args.teacher)

    total = len(tests)
    print(f"\n📊 파일 Top‑1 정확도 : {file_acc:.2f}%\n📊 페이지 Top‑5 정확도 : {page_acc:.2f}%\n")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv("all_cases.csv", index=False, encoding="utf-8-sig")
    if wrong:
        pd.DataFrame(wrong).to_csv("wrong_cases.csv", index=False, encoding="utf-8-sig")
        print("Wrong cases saved → wrong_cases.csv")
    print("All cases saved → all_cases.csv")

if __name__ == "__main__":
    main()

