import cv2
import numpy as np
import cupy as cp  # <-- MUDANÇA GPU: Importamos cupy como cp
import glob
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. CONFIGURAÇÕES PRINCIPAIS ---
CAMINHO_PASTA_ENTRADA = r"C:\Arquivos\Videos\Phantom\v9.1_FNN_Y20250816H004951.195388000_UTC"
NUM_FRAMES_BACKGROUND = 75
NUMERO_DE_CORTES_VERTICAL = 336
plt.style.use('dark_background')

# --- 2. FUNÇÕES ESSENCIAIS (Adaptadas para GPU) ---

def calcular_vetor_referencia_vertical_gpu(arquivos_img, num_frames_bg, num_cortes_v):
    print(f"FASE 1 (GPU): Iniciando calibração com os primeiros {num_frames_bg} frames...")
    if len(arquivos_img) < num_frames_bg:
        raise ValueError(f"Frames insuficientes para calibração.")
    
    # Inicia o vetor de soma na GPU
    soma_medias_v_gpu = cp.zeros(num_cortes_v, dtype=cp.float32) # <-- MUDANÇA GPU
    
    for i, caminho_frame in enumerate(arquivos_img[:num_frames_bg]):
        print(f"  Lendo frame de background {i+1}/{num_frames_bg}...", end='\r')
        # Imagem é lida pela CPU
        img_cinza_cpu = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza_cpu is None: continue
        
        # 1. Transfere a imagem da CPU para a memória da GPU
        img_cinza_gpu = cp.asarray(img_cinza_cpu, dtype=cp.float32) # <-- MUDANÇA GPU
        
        # 2. Todos os cálculos agora são feitos na GPU com CuPy
        cortes_v_gpu = cp.array_split(img_cinza_gpu, num_cortes_v, axis=1) # <-- MUDANÇA GPU
        medias_gpu = cp.array([cp.mean(corte) for corte in cortes_v_gpu]) # <-- MUDANÇA GPU
        soma_medias_v_gpu += medias_gpu
        
    print("\nCalibração na GPU concluída.")
    return soma_medias_v_gpu / num_frames_bg

def analisar_frame_vertical_gpu(img_path, vetor_ref_v_gpu, num_cortes_v):
    img_cinza_cpu = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza_cpu is None: return None
    
    # 1. Transfere a imagem da CPU para a GPU
    img_cinza_gpu = cp.asarray(img_cinza_cpu, dtype=cp.float32) # <-- MUDANÇA GPU

    # 2. Cálculos na GPU
    cortes_v_gpu = cp.array_split(img_cinza_gpu, num_cortes_v, axis=1) # <-- MUDANÇA GPU
    medias_abs_v_gpu = cp.array([cp.mean(corte) for corte in cortes_v_gpu]) # <-- MUDANÇA GPU
    lums_rel_v_gpu = medias_abs_v_gpu - vetor_ref_v_gpu # <-- MUDANÇA GPU
    
    return lums_rel_v_gpu

def plotar_grafico_3d(all_lums_rel_v_cpu, num_cortes_v, total_frames):
    # A função de plot permanece a mesma, recebendo dados da CPU
    print("\nFASE 3: Preparando dados para visualização 3D...")
    Z = np.array(all_lums_rel_v_cpu)
    x = np.arange(num_cortes_v)
    y = np.arange(total_frames)
    X, Y = np.meshgrid(x, y)
    
    if Z.shape != X.shape:
        raise ValueError(f"Inconsistência de dimensões Z:{Z.shape} vs XY:{X.shape}")

    print("Renderizando o gráfico 3D...")
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=4, cstride=4)
    ax.set_xlabel('Espaço (Corte Vertical)', labelpad=15)
    ax.set_ylabel('Tempo (Frame)', labelpad=15)
    ax.set_zlabel('Luminosidade Relativa', labelpad=15)
    ax.set_title('Evolução da Luminosidade Relativa (Processado na GPU)', fontsize=16, pad=20)
    fig.colorbar(surf, shrink=0.6, aspect=10, label='Amplitude')
    ax.view_init(elev=30, azim=-120)
    plt.show()

# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    try:
        arquivos_img = sorted(glob.glob(os.path.join(CAMINHO_PASTA_ENTRADA, '*.jpg')))
        total_frames = len(arquivos_img)
        if total_frames == 0:
            raise FileNotFoundError(f"Nenhum arquivo .jpg encontrado em '{CAMINHO_PASTA_ENTRADA}'")
        print(f"{total_frames} frames encontrados.")

        start_time = time.time()
        
        # ETAPA 1: Calibração na GPU
        vetor_ref_v_gpu = calcular_vetor_referencia_vertical_gpu(arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL)
        
        # ETAPA 2: Coleta de dados na GPU
        print(f"\nFASE 2 (GPU): Coletando métricas de todos os {total_frames} frames...")
        all_lums_rel_vectors_gpu = []
        for i, img_path in enumerate(arquivos_img):
            print(f"  Processando frame {i+1}/{total_frames}...", end='\r')
            lums_rel_v_gpu = analisar_frame_vertical_gpu(img_path, vetor_ref_v_gpu, NUMERO_DE_CORTES_VERTICAL)
            if lums_rel_v_gpu is not None:
                all_lums_rel_vectors_gpu.append(lums_rel_v_gpu)
        print("\nColeta de métricas na GPU concluída.")
        
        # Antes de plotar, precisamos trazer os dados de volta da GPU para a CPU
        print("Transferindo dados da GPU para a CPU para o plot...")
        all_lums_rel_vectors_cpu = [cp.asnumpy(vec) for vec in all_lums_rel_vectors_gpu] # <-- MUDANÇA GPU

        processing_time = time.time() - start_time
        print(f"Tempo de processamento na GPU: {processing_time:.2f} segundos.")
        
        # ETAPA 3: Plotar os dados que estão na CPU
        plotar_grafico_3d(all_lums_rel_vectors_cpu, NUMERO_DE_CORTES_VERTICAL, total_frames)
        
        print("\nProcesso finalizado.")

    except Exception as e:
        print(f"\nERRO: {e}")