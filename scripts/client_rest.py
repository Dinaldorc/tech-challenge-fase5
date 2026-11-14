"""Cliente simples que consome o endpoint de Model Serving via REST.

Uso (credenciais SÓ por variável de ambiente, nunca no código):
    set DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
    set DATABRICKS_TOKEN=<token pessoal>
    python scripts/client_rest.py
"""
import json
import os
import sys

import requests

ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "passos-magicos-risco")

EXEMPLOS = [
    {"inde": 6.8, "iaa": 8.5, "ieg": 7.0, "ips": 6.9, "ipp": 7.2, "ida": 5.5, "ipv": 7.1, "ian": 5.0,
     "fase": 3, "idade": 12, "anos_pm": 2, "defasagem": -1},
    {"inde": 42.0, "ieg": 7.0, "fase": 3},  # inválido: INDE fora da faixa [0, 10]
]


def main() -> int:
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        print("Defina DATABRICKS_HOST e DATABRICKS_TOKEN.", file=sys.stderr)
        return 1
    url = f"{host.rstrip('/')}/serving-endpoints/{ENDPOINT}/invocations"
    resp = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                         json={"dataframe_records": EXEMPLOS}, timeout=60)
    if resp.status_code != 200:
        print(f"Erro HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        return 2
    for entrada, saida in zip(EXEMPLOS, resp.json()["predictions"]):
        print("\nEntrada:", json.dumps(entrada, ensure_ascii=False))
        if saida["status"] == "erro":
            print("  Entrada rejeitada:", saida["erros"])
            continue
        print(f"  Probabilidade estimada: {saida['probabilidade']:.0%} (faixa {saida['faixa']})")
        print(f"  Sinal de apoio: {saida['sinal_apoio']} (limiar {saida['limiar']:.2f})")
        print(f"  {saida['aviso']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
