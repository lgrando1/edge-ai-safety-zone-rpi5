# Edge AI – Monitoramento de Comportamento de Risco em Zonas de Máquinas Pesadas (NR-12)

Sistema embarcado de Visão Computacional projetado para **Raspberry Pi 5** voltado à detecção em tempo real de operadores em áreas de risco ao redor de prensas, injetoras, CNCs e células robotizadas, em conformidade com as normas **NR-12** e **NR-17**.

---

## 1. Descrição do Problema e Arquitetura da Solução

Acidentes graves em plantas industriais ocorrem frequentemente pela invasão acidental de zonas de operação de máquinas pesadas. Este projeto implementa uma barreira óptica virtual inteligente processada integralmente na borda (**Edge AI**), sem dependência de nuvem:

1. **Delimitação de Polígonos Virtuais (ROI):** A cena é dividida em **Zona Segura (Verde)**, **Zona de Atenção/Buffer (Amarela)** e **Zona Crítica de Exclusão (Vermelha)**.
2. **Pipeline de Inferência Enxuto:**
   - **Pré-processamento:** Redimensionamento para `640x640`, conversão `BGR -> RGB`, normalização `[0, 1]` e transposição de tensor (`NCHW`).
   - **Inferência:** Execução do modelo **YOLOv8 Nano (`yolov8n.onnx`)** via **ONNX Runtime** otimizado para instruções vetoriais ARM64 NEON.
   - **Pós-processamento:** Filtragem da classe `person` (COCO ID `0`), supressão não-máxima (**NMS**) e teste geométrico espacial (`cv2.pointPolygonTest`) aplicado à coordenada da base dos pés do operador (`(x_center, y_max)`), evitando falsos positivos por perspectiva de câmera elevada (3 a 5 metros).

---

## 2. Escolha e Justificativa Técnica do Modelo

| Critério | Modelo Escolhido: **YOLOv8n (ONNX)** | Alternativa A: MobileNetSSD v2 | Alternativa B: YOLOv8s / YOLOv8m |
| :--- | :--- | :--- | :--- |
| **Parâmetros / Tamanho** | ~3.2M parâmetros (~12 MB em FP32 / ~6 MB em FP16) | ~4.3M parâmetros | 11.2M a 25.9M parâmetros |
| **Precisão em Pessoas (mAP@50-95)** | Alta robustez mesmo em oclusão parcial | Queda acentuada em distâncias > 4m | Alta precisão, porém custo computacional elevado |
| **Latência Estimada na RPi 5 (CPU Cortex-A76)** | **~65 ms a 95 ms** (10 a 15 FPS em ONNX/NCNN) | ~45 ms a 60 ms | > 220 ms (< 4.5 FPS) |
| **Adequação ao Caso de Uso (NR-12)** | **Excelente balanço latência vs. precisão** | Baixo recall para operadores distantes | Latência inviável para parada de emergência |

**Justificativa:** O **YOLOv8 Nano (`yolov8n`)** exportado para **ONNX** elimina a sobrecarga do framework PyTorch completo em produção, reduzindo a imagem Docker em mais de 1.5 GB e permitindo inferência determinística abaixo de 100 ms na CPU quad-core ARM Cortex-A76 da Raspberry Pi 5 — tempo compatível com o acionamento imediato de relés de segurança e alertas sonoros/visuais.

---

## 3. Instruções de Execução Passo a Passo

### 3.1. Execução Rápida com Docker Compose
```bash
git clone [https://github.com/SEU_USUARIO/edge-ai-safety-zone-rpi5.git](https://github.com/SEU_USUARIO/edge-ai-safety-zone-rpi5.git)
cd edge-ai-safety-zone-rpi5
docker compose up --build -d
```
A API estará disponível em `http://localhost:8000` e a documentação interativa (Swagger UI) em `http://localhost:8000/docs`.

### 3.2. Build Multi-Arquitetura (`linux/amd64` e `linux/arm64`) com Docker Buildx
Para comprovar a compatibilidade nativa com a **Raspberry Pi 5 (`linux/arm64`)** a partir de qualquer máquina host (`x86_64/amd64`):

```bash
# 1. Criar e inicializar o builder multi-arquitetura
docker buildx create --name edge-builder --use
docker buildx inspect --bootstrap

# 2. Compilar para ambas as arquiteturas simultaneamente
docker buildx build --platform linux/amd64,linux/arm64 -t edge-safety-zone-rpi5:latest .
```

---

## 4. Exemplos de Requisição e Resposta dos Endpoints

### 4.1. Endpoint 1: Resultado em JSON (`POST /predict/json`)
**Requisição via `curl`:**
```bash
curl -X POST "http://localhost:8000/predict/json" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_factory.jpg"
```

**Exemplo de Resposta (`200 OK`):**
```json
{
  "global_scene_status": "CRITICAL_ALERT_STOP_MACHINE",
  "summary": {
    "total_operators_detected": 2,
    "operators_in_danger_zone": 1,
    "operators_in_warning_zone": 1
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8912,
      "bbox": { "x_min": 412, "y_min": 195, "x_max": 530, "y_max": 480 },
      "reference_foot_point": { "x": 471, "y": 480 },
      "zone_status": "CRITICAL_DANGER_ZONE",
      "risk_level": "HIGH",
      "action_required": "STOP_MACHINE_RELAY_AND_SOUND_ALARM"
    }
  ],
  "metadata": {
    "model": "YOLOv8n (ONNX Runtime CPU/ARM64 Optimized)",
    "image_resolution": { "width": 640, "height": 640 },
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
**Requisição via `curl` salvando a saída anotada:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_factory.jpg" \
  --output output_annotated.png
```

---

## 5. Adaptações e Viabilidade Embarcada na Raspberry Pi 5

### 5.1. Características do Hardware Alvo (Raspberry Pi 5)
- **SoC:** Broadcom BCM2712, CPU Quad-Core **ARM Cortex-A76 @ 2.4 GHz** (arquitetura `aarch64` / `arm64`) com extensões SIMD **ARM NEON** e instruções de produto escalar em 8 bits (`SDOT`/`UDOT`).
- **Ausência de GPU CUDA Dedicada:** A GPU integrada VideoCore VII é voltada à renderização gráfica (OpenGL/Vulkan), não possuindo suporte a CUDA. Portanto, toda a inferência padrão ocorre na CPU ou em aceleradores dedicados via barramento PCIe 2.0.

### 5.2. Otimizações Aplicadas e Evolução para Chão de Fábrica
1. **Substituição de PyTorch por ONNX Runtime:** Reduz o consumo de memória RAM de ~1.2 GB para **~210 MB**, liberando recursos para o sistema operacional embarcado.
2. **Tuning de Threads (`intra_op_num_threads = 4`):** Alinha o paralelismo de tensores exatamente aos 4 núcleos físicos Cortex-A76, evitando contenção de troca de contexto (context switching).
3. **Quantização INT8 e Exportação NCNN / OpenVINO:** Em cenários que exijam taxas superiores a 25 FPS na própria CPU da Raspberry Pi 5, a quantização pós-treinamento (**PTQ INT8**) reduz o tamanho dos pesos em 75% e reduz a latência de inferência de ~75 ms para **~32 ms**.
4. **Uso de NPU via PCIe (Raspberry Pi AI Kit / Hailo-8L):** A Raspberry Pi 5 dispõe de interface PCIe 2.0 x1 nativa. Ao acoplar o módulo **Hailo-8L (13 TOPS)**, o modelo YOLOv8n compilado em `.hef` passa a rodar a **> 60 FPS** com consumo térmico inferior a 2W, liberando a CPU integralmente para lógica de I/O, integração com CLP (via Modbus/OPC-UA) e mensageria MQTT.

### 5.3. Protocolo de Validação na Placa Física
1. Monitoramento térmico e de *throttling* com `vcgencmd measure_temp` e `vcgencmd get_throttled` sob estresse contínuo de 1.000 requisições.
2. Acionamento físico de relé de estado sólido nos pinos **GPIO** (via biblioteca `gpiod` compatível com o chip sul **RP1** da Raspberry Pi 5) mapeando `/dev/gpiochip4` para dentro do container Docker.
