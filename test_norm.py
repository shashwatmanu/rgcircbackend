from recon_service import normalize_to_xlsx, detect_tpa_choice, parse_mis_universal
f = "/Users/apple/Downloads/raw_mis_data (16) (1).xlsx"
print("Normalizing...")
m_norm = normalize_to_xlsx(f)
print("Normalized to:", m_norm)
t = detect_tpa_choice(m_norm)
print("TPA detected:", t)
parse_mis_universal(m_norm, t)
print("Parse success!")
