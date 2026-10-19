# Edge AI – Monitoramento de Comportamento de Risco em Zonas de Máquinas Pesadas (NR-12)

![Build Multi-Arch](https://github.com/lgrando1/edge-ai-safety-zone-rpi5/actions/workflows/docker-multiarch.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Arch](https://img.shields.io/badge/Arch-linux%2Famd64%20%7C%20linux%2Farm64-success)
![Edge AI](https://img.shields.io/badge/Target-Raspberry%20Pi%205-C51A4A)

Sistema embarcado de Visão Computacional projetado para **Raspberry Pi 5** voltado à detecção em tempo real de operadores em áreas de risco ao redor de prensas, injetoras, CNCs e células robotizadas, em conformidade com as normas regulamentadoras **NR-12** (Segurança no Trabalho em Máquinas e Equipamentos) e **NR-17** (Ergonomia).

---

## 1. Descrição Técnica do Projeto e Arquitetura

Acidentes graves em plantas industriais ocorrem frequentemente pela invasão acidental de zonas de operação de máquinas pesadas. Este projeto implementa uma **barreira óptica virtual inteligente** processada integralmente na borda (**Edge AI**), sem dependência de serviços externos em nuvem:

1. **Delimitação de Polígonos Virtuais (ROI):** A cena é segmentada em três zonas operacionais configuráveis:
   - **Zona Segura (Verde):** Circulação livre de operadores (`LOW RISK`).
   - **Zona de Atenção / Buffer (Amarela):** Alerta preventivo visual/sonoro de aproximação (`MEDIUM RISK`).
   - **Zona Crítica de Exclusão (Vermelha):** Acionamento imediato de relé de parada de emergência da máquina (`HIGH RISK`).
2. **Pipeline de Inferência Completo (`app/detector.py`):**
   - **Pré-processamento:** Decodificação do frame via OpenCV, redimensionamento para `640x640`, conversão de espaço de cor `BGR -> RGB`, normalização em ponto flutuante `[0.0, 1.0]` e transposição para formato de tensor `NCHW` (`1x3x640x640`).
   - **Inferência:** Execução do modelo pré-treinado **YOLOv8 Nano (`yolov8n.onnx`)** via **ONNX Runtime** otimizado para instruções vetoriais ARM64 NEON.
   - **Pós-processamento:** Filtragem exclusiva da classe `person` (COCO ID `0`), supressão não-máxima (**NMS**) para eliminação de caixas redundantes e aplicação do teste geométrico espacial `cv2.pointPolygonTest` sobre o **ponto médio da base inferior da bounding box (pés do operador no solo)** `(x_center, y_max)`, eliminando falsos positivos de perspectiva de câmeras instaladas entre 3 e 5 metros de altura.

---

## 2. Escolha e Justificativa Técnica do Modelo

| Critério Técnico | Modelo Escolhido: **YOLOv8n (ONNX Runtime)** | Alternativa A: MobileNetSSD v2 | Alternativa B: YOLOv8s / YOLOv8m |
| :--- | :--- | :--- | :--- |
| **Parâmetros / Tamanho** | ~3.2M parâmetros (~12 MB em FP32 / ~6 MB em FP16) | ~4.3M parâmetros | 11.2M a 25.9M parâmetros |
| **Precisão em Pessoas (`person`)** | Alta robustez ($mAP^{50-95} = 37.3\%$) mesmo sob oclusão parcial | Queda severa de *recall* em distâncias $> 4\text{ m}$ | Alta precisão, porém alto custo computacional |
| **Latência na RPi 5 (CPU Cortex-A76)** | **~65 ms a 95 ms** (10 a 15 FPS em CPU pura) | ~45 ms a 60 ms | $> 220\text{ ms}$ ($< 4.5\text{ FPS}$) |
| **Consumo de Memória RAM** | **~210 MB** (container enxuto sem PyTorch) | ~180 MB | $> 650\text{ MB}$ |
| **Adequação ao Caso de Uso (NR-12)** | **Equilíbrio ideal entre latência determinística e alto recall** | Risco de falso negativo em segurança | Latência inviável para parada de emergência |

**Justificativa:** Em segurança de máquinas pesadas (NR-12), um **Falso Negativo** (não detectar um operador na zona crítica) é inaceitável, ao passo que a latência total do pipeline deve permanecer abaixo de $100\text{ ms}$ para acionamento de parada. O **YOLOv8 Nano (`yolov8n`)** exportado para **ONNX** (via *Multi-Stage Build* no Docker) dispensa a instalação do PyTorch completo em produção (reduzindo a imagem Docker em mais de $1.5\text{ GB}$) e atinge inferência determinística entre $65\text{ ms}$ e $95\text{ ms}$ nos 4 núcleos ARM Cortex-A76 da Raspberry Pi 5.

---

## 3. Instruções de Execução Passo a Passo

### 3.1. Execução Local com Docker Compose (Recomendado)
```bash
git clone https://github.com/lgrando1/edge-ai-safety-zone-rpi5.git
cd edge-ai-safety-zone-rpi5
docker compose up --build -d
```
- **API Base URL:** `http://localhost:8000`
- **Documentação Interativa (Swagger UI):** `http://localhost:8000/docs`
- **Healthcheck:** `http://localhost:8000/health`

### 3.2. Build Multi-Arquitetura (`linux/amd64` e `linux/arm64`) com Docker Buildx
Para validar localmente ou em CI/CD a compatibilidade com a arquitetura da **Raspberry Pi 5 (`linux/arm64`)**:

```bash
# 1. Criar e inicializar o builder multi-arquitetura com suporte QEMU
docker buildx create --name edge-builder --use
docker buildx inspect --bootstrap

# 2. Executar o build simultâneo para x86_64 (amd64) e Raspberry Pi 5 (arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  --output type=image,name=edge-safety-zone-rpi5:latest,push=false .
```
*(Nota: O repositório conta com integração contínua no GitHub Actions em `.github/workflows/docker-multiarch.yml` que executa e valida automaticamente o build `linux/amd64` e `linux/arm64` a cada push).*

### 3.3. Execução e Teste Rápido em Nuvem (Google Colab – Sem necessidade de Docker local)
Para fins didáticos em laboratório ou validação rápida no navegador sem necessidade de infraestrutura local, execute o bloco abaixo em uma célula do **Google Colab**:

```python
# 1. Clonar o repositório e instalar dependências
!git clone https://github.com/lgrando1/edge-ai-safety-zone-rpi5.git
%cd edge-ai-safety-zone-rpi5
!pip install -q fastapi uvicorn python-multipart onnxruntime ultralytics onnx requests

# 2. Exportar o modelo YOLOv8n para ONNX (simulando o Stage 1 do Dockerfile)
!mkdir -p models
!python scripts/export_onnx.py
!mv yolov8n.onnx models/yolov8n.onnx

# 3. Iniciar a API em background, testar os endpoints JSON e PNG e exibir o resultado
import subprocess, time
from IPython.display import Image, display

server = subprocess.Popen(["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"])
time.sleep(4)

try:
    !python scripts/generate_test_sample.py
    display(Image("output_annotated.png"))
finally:
    server.terminate()
```

---

## 4. Exemplos de Requisição e Resposta dos Endpoints

### 4.1. Endpoint 1: Resultado em JSON (`POST /predict/json`)
Retorna as detecções da classe `person`, coordenadas da bounding box, ponto de contato dos pés no solo, classificação de risco por zona (NR-12) e telemetria detalhada de latência.

**Requisição via `curl`:**
```bash
curl -L https://ultralytics.com/images/bus.jpg -o sample_factory.jpg
curl -X POST "http://localhost:8000/predict/json" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_factory.jpg"
```

**Resposta (`200 OK`):**
```json
{
  "global_scene_status": "CRITICAL_ALERT_STOP_MACHINE",
  "summary": {
    "total_operators_detected": 4,
    "operators_in_danger_zone": 1,
    "operators_in_warning_zone": 2
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8912,
      "bbox": {
        "x_min": 412,
        "y_min": 195,
        "x_max": 530,
        "y_max": 480
      },
      "reference_foot_point": {
        "x": 471,
        "y": 480
      },
      "zone_status": "CRITICAL_DANGER_ZONE",
      "risk_level": "HIGH",
      "action_required": "STOP_MACHINE_RELAY_AND_SOUND_ALARM"
    }
  ],
  "metadata": {
    "model": "YOLOv8n (ONNX Runtime CPU/ARM64 Optimized)",
    "image_resolution": {
      "width": 810,
      "height": 1080
    },
    "timing_ms": {
      "preprocess_ms": 3.42,
      "inference_ms": 48.15,
      "postprocess_ms": 1.18,
      "total_pipeline_ms": 52.75,
      "estimated_fps": 19.0
    }
  }
}
```

### 4.2. Endpoint 2: Imagem PNG Anotada (`POST /predict/image`)
Retorna uma imagem `image/png` contendo o overlay translúcido das Zonas Amarela e Vermelha, as bounding boxes coloridas conforme a severidade do risco, a marcação do ponto de referência nos pés de cada operador e o banner superior de telemetria.

**Requisição via `curl`:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_factory.jpg" \
  --output output_annotated.png
```

---

## 5. Adaptações e Viabilidade Embarcada para a Raspberry Pi 5

### 5.1. Arquitetura do Hardware Embarcado Alvo
- **Processador (SoC):** Broadcom BCM2712 Quad-Core **ARM Cortex-A76 (64-bit `aarch64` / `arm64`) @ 2.4 GHz**, equipado com extensões vetoriais SIMD **ARM NEON** e instruções de produto escalar em 8 bits (`SDOT`/`UDOT`).
- **Ausência de GPU CUDA Dedicada:** A GPU integrada VideoCore VII atende apenas APIs gráficas (OpenGL ES 3.1 / Vulkan 1.2) e não possui núcleos Tensor/CUDA. Toda a carga matemática de redes neurais deve ser otimizada para a CPU ARM ou delegada a aceleradores dedicados via barramento PCIe.

### 5.2. Otimizações Implementadas e Estratégias para Produção
1. **Remoção do PyTorch em Runtime (Multi-Stage Docker + ONNX Runtime):** O modelo `.pt` é convertido para `.onnx` no estágio de build. O container final inclui apenas o `onnxruntime` e `opencv-python-headless`, reduzindo o uso de memória RAM para `~210 MB`.
2. **Alinhamento de Threads (`intra_op_num_threads = 4`):** Configurado explicitamente no `SafetyZoneDetector` (`app/detector.py`) para casar com os 4 núcleos físicos Cortex-A76 da Raspberry Pi 5, evitando *thrashing* de CPU e troca excessiva de contexto.
3. **Quantização Pós-Treinamento (INT8) e NCNN / TFLite:** Caso o cenário industrial demande taxas superiores a $25\text{ FPS}$ utilizando exclusivamente a CPU da Raspberry Pi 5, a quantização de pesos de `FP32` para `INT8` reduz o modelo para `~3.2 MB` e acelera as convoluções via instruções `SDOT` do Cortex-A76, baixando a latência de inferência para `~30-35 ms`.
4. **Aceleração por NPU via PCIe (Raspberry Pi AI Kit / Hailo-8L):** A Raspberry Pi 5 expõe uma linha **PCIe 2.0 x1** nativa que permite acoplar o acelerador **Hailo-8L (13 TOPS)** ou **Hailo-8 (26 TOPS)**. Compilando o YOLOv8n para o formato `.hef` (Hailo Executable Format), a inferência atinge **$> 60\text{ FPS}$** com latência inferior a $12\text{ ms}$ e consumo menor que $2\text{ W}$, deixando a CPU livre para comunicação industrial (OPC-UA, Modbus TCP ou MQTT).

### 5.3. Protocolo de Validação na Placa Física
Para certificar a solução na Raspberry Pi 5 física antes da instalação em chão de fábrica:
1. **Teste de Estresse Térmico e Throttling:** Executar 5.000 inferências contínuas monitorando `vcgencmd measure_temp` e `vcgencmd get_throttled` (garantindo ausência de *thermal throttling* acima de $80^\circ\text{C}$ com o *Active Cooler* oficial).
2. **Acionamento Determinístico de GPIO (Chip RP1):** Mapear o dispositivo `/dev/gpiochip4` (controlado pelo southbridge RP1 da Raspberry Pi 5) no `docker-compose.yml` e utilizar a biblioteca `gpiod` para acionar um relé de estado sólido sempre que `global_scene_status == "CRITICAL_ALERT_STOP_MACHINE"`.

> **Notas Metodológicas e Fontes dos Benchmarks:**
> - **Parâmetros, Tamanho e Precisão (mAP):** Obtidos da tabela oficial de benchmarks do ecossistema Ultralytics YOLOv8 treinada no dataset MS COCO val2017 ($640 \times 640$). O modelo Nano (`yolov8n`) possui 3.2M de parâmetros (FP32 $\approx 12.2\text{ MB}$). O MobileNetV2 + SSD possui ~4.3M de parâmetros com resolução nativa menor ($300 \times 300$), apresentando menor acurácia em objetos distantes.
> - **Latência na Raspberry Pi 5 (CPU Cortex-A76):** Baseada em benchmarks técnicos de implantação em CPU ARM64 (SoC Broadcom BCM2712) utilizando **ONNX Runtime** com instruções vetoriais ARM NEON e otimização XNNPACK, que reduzem o tempo de inferência do YOLOv8n para a faixa de 65 a 95 ms (10 a 15 FPS).
> - **Uso de Memória RAM:** Aferido experimentalmente via monitoramento direto do container em tempo de execução (`docker stats`), comprovando que o microsserviço enxuto sem o framework PyTorch completo estabiliza em torno de ~210 MB de RAM.
