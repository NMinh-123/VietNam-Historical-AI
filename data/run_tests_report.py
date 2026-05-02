#!/usr/bin/env python3
"""
run_tests_report.py — Chạy toàn bộ test suite và xuất báo cáo số liệu.
======================================================================
Xuất ra:
  - Terminal: bảng màu tóm tắt theo nhóm
  - File JSON: test_report_<timestamp>.json  (dùng cho báo cáo sau)
  - File TXT : test_report_<timestamp>.txt   (dùng cho báo cáo sau)

Cách chạy:
    cd "VietNam Historical AI"
    python3 data/run_tests_report.py

Yêu cầu:
    - Django project nằm ở frontend/
    - Đã seed timeline: python3 data/seed_timeline.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"
OUT_DIR  = ROOT / "data" / "test_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Màu terminal ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Nhóm test ─────────────────────────────────────────────────────────────────
GROUPS = {
    "A. Models":       "DynastyModelTests|KingModelTests",
    "B. Views HTTP":   "HomeViewTests|TimelineViewTests",
    "C. Persona":      "PersonaViewTests",
    "D. Timeline Seed":"TimelineDataTests",
    "E. API Proxy":    "PersonaChatApiTests",
    "F. Guardrail":    "GuardrailLogicTests",
    "G. Regression":   "UrlRegressionTests",
}


def run_tests() -> tuple[str, float]:
    """Chạy manage.py test và trả về (output, elapsed_seconds)."""
    print(f"{CYAN}{BOLD}▶ Đang chạy test suite...{RESET}\n")
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "manage.py", "test", "core", "--verbosity=2"],
        cwd=FRONTEND,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    # Django test output goes to stderr
    output = result.stderr + result.stdout
    return output, elapsed


def parse_output(output: str) -> dict:
    """Phân tích output của Django test runner."""
    tests = []

    # Mỗi dòng test: "test_xyz (core.tests.ClassName.test_xyz) ... ok/FAIL/ERROR/skipped"
    pattern = re.compile(
        r'^(test_\S+)\s+\(core\.tests\.(\w+)\.(test_\S+)\)\s*'
        r'(?:.*?)\.\.\.\s*(ok|FAIL|ERROR|skipped.*?)$',
        re.MULTILINE
    )
    for m in pattern.finditer(output):
        test_name  = m.group(1)
        class_name = m.group(2)
        status_raw = m.group(4).strip()

        if status_raw.startswith('skipped'):
            status = 'SKIP'
            reason = re.sub(r"^skipped '?(.*?)'?$", r'\1', status_raw)
        elif status_raw == 'ok':
            status = 'PASS'
            reason = ''
        else:
            status = status_raw  # FAIL / ERROR
            reason = ''

        tests.append({
            'class':  class_name,
            'name':   test_name,
            'status': status,
            'reason': reason,
        })

    # Fallback: parse summary line
    summary_match = re.search(
        r'Ran (\d+) tests? in ([\d.]+)s\s*\n\s*(\w+)(?:\s+\((.+?)\))?',
        output,
    )
    summary = {}
    if summary_match:
        summary['ran']    = int(summary_match.group(1))
        summary['result'] = summary_match.group(3)  # OK / FAILED
        extra = summary_match.group(4) or ''
        fail_m  = re.search(r'failures=(\d+)', extra)
        err_m   = re.search(r'errors=(\d+)',   extra)
        skip_m  = re.search(r'skipped=(\d+)',  extra)
        summary['failures'] = int(fail_m.group(1)) if fail_m else 0
        summary['errors']   = int(err_m.group(1))  if err_m  else 0
        summary['skipped']  = int(skip_m.group(1)) if skip_m else 0
        summary['passed']   = (summary['ran']
                               - summary['failures']
                               - summary['errors']
                               - summary['skipped'])

    return {'tests': tests, 'summary': summary}


def group_results(tests: list[dict]) -> dict:
    """Nhóm kết quả theo GROUPS."""
    grouped = {}
    for group_label, pattern in GROUPS.items():
        regex = re.compile(pattern)
        group_tests = [t for t in tests if regex.search(t['class'])]
        counts = {
            'pass':  sum(1 for t in group_tests if t['status'] == 'PASS'),
            'fail':  sum(1 for t in group_tests if t['status'] == 'FAIL'),
            'error': sum(1 for t in group_tests if t['status'] == 'ERROR'),
            'skip':  sum(1 for t in group_tests if t['status'] == 'SKIP'),
            'total': len(group_tests),
        }
        grouped[group_label] = {'counts': counts, 'tests': group_tests}
    return grouped


def print_report(grouped: dict, summary: dict, elapsed: float):
    """In báo cáo màu ra terminal."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    width = 72

    print()
    print(f"{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD}  VICAL AI — BÁO CÁO KIỂM THỬ                  {now}{RESET}")
    print(f"{BOLD}{'═' * width}{RESET}")

    # Header bảng
    print(f"\n  {'Nhóm':<32} {'Pass':>5} {'Fail':>5} {'Skip':>5} {'Total':>6}  Trạng thái")
    print(f"  {'─' * 32} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}  {'─' * 14}")

    total_pass = total_fail = total_error = total_skip = total_all = 0

    for group_label, data in grouped.items():
        c = data['counts']
        total_pass  += c['pass']
        total_fail  += c['fail'] + c['error']
        total_skip  += c['skip']
        total_all   += c['total']

        has_fail = (c['fail'] + c['error']) > 0
        has_skip = c['skip'] > 0 and (c['fail'] + c['error']) == 0
        all_pass = (c['fail'] + c['error'] + c['skip']) == 0 and c['total'] > 0

        if all_pass:
            state_str = f"{GREEN}✅ PASS{RESET}"
        elif has_fail:
            state_str = f"{RED}❌ FAIL{RESET}"
        elif has_skip:
            state_str = f"{YELLOW}⏭  SKIP{RESET}"
        else:
            state_str = f"{DIM}── empty{RESET}"

        pass_col  = f"{GREEN}{c['pass']:>5}{RESET}"  if c['pass']  else f"{DIM}{c['pass']:>5}{RESET}"
        fail_col  = f"{RED}{c['fail']+c['error']:>5}{RESET}" if c['fail']+c['error'] else f"{DIM}    0{RESET}"
        skip_col  = f"{YELLOW}{c['skip']:>5}{RESET}" if c['skip']  else f"{DIM}    0{RESET}"
        total_col = f"{BOLD}{c['total']:>6}{RESET}"

        print(f"  {group_label:<32} {pass_col} {fail_col} {skip_col} {total_col}  {state_str}")

    print(f"  {'─' * 32} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}")
    all_good = total_fail == 0
    total_str = f"{GREEN if all_good else RED}{BOLD}  {'TỔNG CỘNG':<32} {total_pass:>5} {total_fail:>5} {total_skip:>5} {total_all:>6}{RESET}"
    print(total_str)

    # Tóm tắt
    print(f"\n  ⏱  Thời gian chạy : {elapsed:.2f}s")
    rate = round(total_pass / total_all * 100, 1) if total_all else 0
    print(f"  📊 Tỷ lệ Pass      : {BOLD}{rate}%{RESET}  ({total_pass}/{total_all - total_skip} tests hoạt động)")
    print(f"  📦 Tổng tests      : {total_all}  |  Skip (cần seed): {total_skip}")

    overall = f"{GREEN}{BOLD}✅  TẤT CẢ PASS{RESET}" if all_good else f"{RED}{BOLD}❌  CÓ LỖI{RESET}"
    print(f"\n  Kết quả tổng thể  : {overall}")
    print(f"{BOLD}{'═' * width}{RESET}\n")

    # Chi tiết FAIL/ERROR
    failed_tests = [
        t for grp in grouped.values()
        for t in grp['tests']
        if t['status'] in ('FAIL', 'ERROR')
    ]
    if failed_tests:
        print(f"{RED}{BOLD}  Tests thất bại:{RESET}")
        for t in failed_tests:
            print(f"    {RED}✗{RESET} {t['class']}.{t['name']}")
        print()


def save_reports(grouped: dict, summary: dict, elapsed: float, raw_output: str):
    """Lưu báo cáo ra file JSON và TXT."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── JSON ─────────────────────────────────────────────────────────────────
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
        "groups": {
            label: {
                "counts": data["counts"],
                "tests": [
                    {"name": t["name"], "status": t["status"], "reason": t["reason"]}
                    for t in data["tests"]
                ],
            }
            for label, data in grouped.items()
        },
    }
    json_path = OUT_DIR / f"test_report_{ts}.json"
    json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding='utf-8')

    # ── TXT ──────────────────────────────────────────────────────────────────
    lines = [
        "VICAL AI — BÁO CÁO KIỂM THỬ",
        f"Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Thời gian chạy: {elapsed:.2f}s",
        "=" * 70,
        "",
        f"{'Nhóm':<32} {'Pass':>5} {'Fail':>5} {'Skip':>5} {'Total':>6}",
        f"{'─' * 32} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}",
    ]
    tot_p = tot_f = tot_s = tot_t = 0
    for label, data in grouped.items():
        c = data['counts']
        lines.append(f"{label:<32} {c['pass']:>5} {c['fail']+c['error']:>5} {c['skip']:>5} {c['total']:>6}")
        tot_p += c['pass']; tot_f += c['fail']+c['error']
        tot_s += c['skip']; tot_t += c['total']
    lines += [
        f"{'─' * 32} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}",
        f"{'TỔNG CỘNG':<32} {tot_p:>5} {tot_f:>5} {tot_s:>5} {tot_t:>6}",
        "",
        f"Tỷ lệ pass: {round(tot_p / (tot_t - tot_s) * 100, 1) if (tot_t - tot_s) else 0}%",
        f"Kết quả: {'PASS' if tot_f == 0 else 'FAIL'}",
        "",
        "─" * 70,
        "CHI TIẾT TỪNG TEST",
        "─" * 70,
    ]
    for label, data in grouped.items():
        lines.append(f"\n[{label}]")
        for t in data['tests']:
            icon = {'PASS': '✓', 'FAIL': '✗', 'ERROR': '!', 'SKIP': '○'}.get(t['status'], '?')
            reason = f"  ({t['reason']})" if t['reason'] else ''
            lines.append(f"  {icon} [{t['status']:<5}] {t['name']}{reason}")

    lines += ["", "─" * 70, "RAW OUTPUT (Django test runner)", "─" * 70, "", raw_output]

    txt_path = OUT_DIR / f"test_report_{ts}.txt"
    txt_path.write_text('\n'.join(lines), encoding='utf-8')

    return json_path, txt_path


def main():
    print(f"\n{BOLD}{'━' * 72}{RESET}")
    print(f"{BOLD}  Vical AI Test Reporter{RESET}  —  {DIM}{datetime.now().strftime('%d/%m/%Y %H:%M')}{RESET}")
    print(f"{BOLD}{'━' * 72}{RESET}\n")
    print(f"  Project : {FRONTEND}")
    print(f"  Báo cáo : {OUT_DIR}\n")

    raw_output, elapsed = run_tests()
    parsed   = parse_output(raw_output)
    grouped  = group_results(parsed['tests'])
    summary  = parsed.get('summary', {})

    print_report(grouped, summary, elapsed)

    json_path, txt_path = save_reports(grouped, summary, elapsed, raw_output)

    print(f"  {CYAN}📄 JSON: {json_path.name}{RESET}")
    print(f"  {CYAN}📄 TXT : {txt_path.name}{RESET}")
    print(f"  {DIM}Thư mục: {OUT_DIR}{RESET}\n")

    # Exit code
    failed = any(
        t['status'] in ('FAIL', 'ERROR')
        for grp in grouped.values()
        for t in grp['tests']
    )
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
