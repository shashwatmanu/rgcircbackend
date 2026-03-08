import subprocess
import json
import traceback

CMD = [
    "curl", "-s", "-X", "POST", "http://localhost:8000/reconcile/v2/bulk",
    "-H", "Content-Type: multipart/form-data",
    "-F", "bank_files=@/Users/apple/Downloads/bank_1_OpTransactionHistoryUX330-01-2026 (2).xlsx",
    "-F", "mis_files=@/Users/apple/Downloads/raw_mis_data (16) (1).xlsx",
    "-F", "outstanding_file=@/Users/apple/Downloads/Outstanding_300126.xlsx"
]

try:
    res = subprocess.run(CMD, capture_output=True, text=True)
    out = res.stdout
    for line in out.splitlines():
        if not line: continue
        try:
            d = json.loads(line)
            if d.get("type") == "pair_result":
                print("PAIR RESULT:", json.dumps(d, indent=2))
        except:
             pass
except Exception as e:
    traceback.print_exc()
