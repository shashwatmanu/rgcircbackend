import urllib.request
import urllib.parse
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json

m = MultipartEncoder(
    fields={
        'bank_files': ('bank_1_OpTransactionHistoryUX330-01-2026 (2).xlsx', open('/Users/apple/Downloads/bank_1_OpTransactionHistoryUX330-01-2026 (2).xlsx', 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        'mis_files': ('raw_mis_data (16) (1).xlsx', open('/Users/apple/Downloads/raw_mis_data (16) (1).xlsx', 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        'outstanding_file': ('Outstanding_300126.xlsx', open('/Users/apple/Downloads/Outstanding_300126.xlsx', 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    }
)

req = urllib.request.Request(
    'http://localhost:8000/reconcile/v2/bulk', 
    data=m.to_string(),
    headers={'Content-Type': m.content_type}
)

with urllib.request.urlopen(req) as response:
    for line in response:
        if line:
            lstr = line.decode('utf-8').strip()
            print(lstr)
