#!/usr/bin/env python3
"""
Sistema PoA (Power of Arrival) 1D com Simulação
Rastreamento de Foguetes - Cálculo de Distância via RSSI
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

class RFSignalSimulator:
    """Simula captura de sinal RF de um transmissor em movimento"""
    
    def __init__(self, tx_power_dbm=20, freq_hz=433e6, sample_rate=2.048e6):
        """
        tx_power_dbm: Potência do transmissor (HC-12 = 20 dBm)
        freq_hz: Frequência (433 MHz)
        sample_rate: Taxa de amostragem
        """
        self.tx_power_dbm = tx_power_dbm
        self.freq_hz = freq_hz
        self.sample_rate = sample_rate
        self.wavelength = 3e8 / freq_hz
        
        print(f"═" * 60)
        print(f"  RF Signal Simulator - PoA 1D")
        print(f"═" * 60)
        print(f"  Potência TX: {tx_power_dbm} dBm")
        print(f"  Frequência: {freq_hz/1e6:.1f} MHz")
        print(f"  Wavelength: {self.wavelength:.4f} m")
        print(f"  Taxa amostragem: {sample_rate/1e6:.3f} MSps")
        print(f"═" * 60)
    
    def generate_signal(self, distance_m, duration_s=1.0, snr_db=20):
        """
        Gera sinal I/Q com ruído gaussiano
        
        distance_m: Distância do transmissor
        duration_s: Duração da captura
        snr_db: Signal-to-Noise Ratio
        """
        num_samples = int(self.sample_rate * duration_s)
        
        # Sinal de portadora (433 MHz)
        t = np.arange(num_samples) / self.sample_rate
        phase = 2 * np.pi * self.freq_hz * t
        
        # Componentes I/Q
        i_signal = np.cos(phase)
        q_signal = np.sin(phase)
        signal = i_signal + 1j * q_signal
        
        # Path Loss (Friis Free Space)
        path_loss_db = 20 * np.log10(4 * np.pi * distance_m / self.wavelength)
        
        # RSSI teórico (potência recebida)
        rssi_dbm = self.tx_power_dbm - path_loss_db
        
        # Atenuação linear
        attenuation_linear = 10 ** (-path_loss_db / 20)
        signal_attenuated = signal * attenuation_linear
        
        # Ruído
        noise_power = 10 ** ((rssi_dbm - snr_db) / 10)
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 
                                            1j * np.random.randn(num_samples))
        
        # Sinal final
        signal_final = signal_attenuated + noise
        
        return {
            'signal': signal_final,
            'distance_real': distance_m,
            'rssi_dbm_theoretical': rssi_dbm,
            'path_loss_db': path_loss_db,
            'snr_db': snr_db,
            'num_samples': num_samples
        }
    
    def calculate_rssi(self, signal_iq):
        """Calcula RSSI a partir de amostras I/Q"""
        # Potência média do sinal
        power_linear = np.mean(np.abs(signal_iq) ** 2)
        
        # Converter para dBm (referência 1mW = 0dBm)
        # Fórmula: P[dBm] = 10*log10(P[W]) + 30
        # Para sinal normalizado, usamos referência relativa
        rssi_dbm = 10 * np.log10(power_linear + 1e-10)
        
        return rssi_dbm, power_linear
    
    def estimate_distance_poa(self, rssi_dbm_measured):
        """
        Estima distância usando PoA (Power of Arrival)
        
        Fórmula de Path Loss (Friis):
        P_rx[dBm] = P_tx[dBm] - 20*log10(4*pi*d/λ)
        
        Resolvendo para d:
        d = λ/(4*pi) * 10^((P_tx - P_rx)/20)
        """
        path_loss_db = self.tx_power_dbm - rssi_dbm_measured
        distance = (self.wavelength / (4 * np.pi)) * 10 ** (path_loss_db / 20)
        
        return distance, path_loss_db


class PoATracker:
    """Rastreador PoA com buffer circular"""
    
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.distances = []
        self.rssi_values = []
        self.timestamps = []
    
    def add_measurement(self, distance, rssi_dbm, timestamp=None):
        """Adiciona medição ao buffer"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.distances.append(distance)
        self.rssi_values.append(rssi_dbm)
        self.timestamps.append(timestamp)
        
        # Manter tamanho do buffer
        if len(self.distances) > self.buffer_size:
            self.distances.pop(0)
            self.rssi_values.pop(0)
            self.timestamps.pop(0)
    
    def get_smoothed_distance(self):
        """Retorna distância suavizada (média móvel)"""
        if not self.distances:
            return None
        return np.mean(self.distances)
    
    def get_std_deviation(self):
        """Desvio padrão das medições"""
        if len(self.distances) < 2:
            return 0
        return np.std(self.distances)


def test_poa_accuracy():
    """Testa acurácia do sistema PoA com distâncias conhecidas"""
    
    print("\n" + "=" * 60)
    print("TESTE 1: ACURÁCIA PoA COM DISTÂNCIAS CONHECIDAS")
    print("=" * 60 + "\n")
    
    simulator = RFSignalSimulator(tx_power_dbm=20, freq_hz=433e6)
    tracker = PoATracker(buffer_size=5)
    
    # Distâncias a testar
    test_distances = [5, 10, 20, 30, 50, 100]
    results = []
    
    for distance in test_distances:
        print(f"Testando distância: {distance}m")
        
        # Gerar sinal simulado
        signal_data = simulator.generate_signal(
            distance_m=distance,
            duration_s=0.5,
            snr_db=20
        )
        
        # Calcular RSSI
        rssi_dbm, power_linear = simulator.calculate_rssi(signal_data['signal'])
        
        # Estimar distância via PoA
        distance_estimated, path_loss = simulator.estimate_distance_poa(rssi_dbm)
        
        # Adicionar ao tracker
        tracker.add_measurement(distance_estimated, rssi_dbm)
        
        # Calcular erro
        error = abs(distance_estimated - distance)
        error_percent = (error / distance) * 100
        
        result = {
            'distance_real': distance,
            'distance_estimated': round(distance_estimated, 2),
            'rssi_dbm': round(rssi_dbm, 2),
            'rssi_theoretical': round(signal_data['rssi_dbm_theoretical'], 2),
            'error_m': round(error, 2),
            'error_percent': round(error_percent, 2),
            'path_loss_db': round(path_loss, 2)
        }
        
        results.append(result)
        
        print(f"  Real: {distance}m → Estimado: {distance_estimated:.2f}m")
        print(f"  RSSI: {rssi_dbm:.2f} dBm (teórico: {signal_data['rssi_dbm_theoretical']:.2f} dBm)")
        print(f"  Erro: {error:.2f}m ({error_percent:.1f}%)\n")
    
    # Média suavizada
    smoothed = tracker.get_smoothed_distance()
    std_dev = tracker.get_std_deviation()
    
    print(f"Última medição suavizada: {smoothed:.2f}m (σ = {std_dev:.2f}m)")
    
    return results


def test_moving_target():
    """Simula foguete em movimento"""
    
    print("\n" + "=" * 60)
    print("TESTE 2: FOGUETE EM MOVIMENTO (SIMULADO)")
    print("=" * 60 + "\n")
    
    simulator = RFSignalSimulator(tx_power_dbm=20, freq_hz=433e6)
    tracker = PoATracker(buffer_size=8)
    
    # Simular trajetória de foguete (vai de 10m a 50m e volta)
    time_points = np.linspace(0, 4, 20)  # 4 segundos, 20 pontos
    distances_trajectory = 10 + 40*np.abs(np.sin(np.pi * time_points / 4))
    
    measurements = []
    
    print("Tempo(s) | Distância(m) | RSSI(dBm) | Erro(%)")
    print("-" * 50)
    
    for i, distance_real in enumerate(distances_trajectory):
        # Gerar sinal
        signal_data = simulator.generate_signal(
            distance_m=distance_real,
            duration_s=0.2,
            snr_db=15
        )
        
        # Calcular RSSI
        rssi_dbm, _ = simulator.calculate_rssi(signal_data['signal'])
        
        # Estimar distância
        distance_est, _ = simulator.estimate_distance_poa(rssi_dbm)
        
        # Adicionar ao tracker
        tracker.add_measurement(distance_est, rssi_dbm, 
                              timestamp=f"{i*0.2:.1f}s")
        
        error_percent = abs(distance_est - distance_real) / distance_real * 100
        
        measurement = {
            'tempo_s': round(i*0.2, 1),
            'distancia_real': round(distance_real, 2),
            'distancia_estimada': round(distance_est, 2),
            'rssi_dbm': round(rssi_dbm, 2),
            'erro_percent': round(error_percent, 1)
        }
        measurements.append(measurement)
        
        print(f"{i*0.2:>6.1f}   | {distance_real:>11.2f}  | {rssi_dbm:>8.2f} | {error_percent:>6.1f}")
    
    return measurements


def save_results_to_file(test_name, results, output_dir='./test_results'):
    """Salva resultados em JSON"""
    Path(output_dir).mkdir(exist_ok=True)
    
    filename = f"{output_dir}/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Resultados salvos em: {filename}")
    return filename


def main():
    """Executa todos os testes"""
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " SISTEMA DE RASTREAMENTO DE FOGUETES - PoA 1D ".center(58) + "║")
    print("║" + " Simulação com Dados Sintéticos ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Teste 1: Acurácia
    results_test1 = test_poa_accuracy()
    save_results_to_file("test_accuracy", results_test1)
    
    # Teste 2: Movimento
    results_test2 = test_moving_target()
    save_results_to_file("test_movement", results_test2)
    
    print("\n" + "=" * 60)
    print("TESTES CONCLUÍDOS COM SUCESSO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. Verificar os arquivos JSON em ./test_results/")
    print("2. Quando tiver RTL-SDR, adaptar para ler dados reais")
    print("3. Expandir para múltiplas antenas (AoA)")
    print()


if __name__ == '__main__':
    main()
