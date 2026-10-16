import urllib.request
import json
import requests

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = "sample_factory.jpg"
API_BASE = "http://localhost:8000"

def run_demo():
    print("1. Baixando imagem de teste padrao...")
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    print(f"   Imagem salva em: {SAMPLE_PATH}")

    print("\n2. Testando Endpoint JSON (POST /predict/json)...")
    with open(SAMPLE_PATH, "rb") as f:
        resp_json = requests.post(f"{API_BASE}/predict/json", files={"file": ("sample.jpg", f, "image/jpeg")})
    print(f"   Status HTTP: {resp_json.status_code}")
    print(json.dumps(resp_json.json(), indent=2, ensure_ascii=False))

    print("\n3. Testando Endpoint Imagem Anotada (POST /predict/image)...")
    with open(SAMPLE_PATH, "rb") as f:
        resp_img = requests.post(f"{API_BASE}/predict/image", files={"file": ("sample.jpg", f, "image/jpeg")})
    print(f"   Status HTTP: {resp_img.status_code}")
    if resp_img.status_code == 200:
        out_file = "output_annotated.png"
        with open(out_file, "wb") as out:
            out.write(resp_img.content)
        print(f"   Imagem PNG anotada salva com sucesso em: {out_file}")

if __name__ == "__main__":
    run_demo()
