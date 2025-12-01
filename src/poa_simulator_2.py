#!/usr/bin/env python3
"""
Sistema PoA 1D - CALIBRAÇÃO AUTOMÁTICA
Ajusta fator de escala baseado na primeira medição conhecida
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class RFSignalSimulator:
    """Simulador com calibração automática"""
    
    def __init__(self, tx_power_dbm=20, freq_hz=433e6, sample_rate=2.048e6):
        self.tx_power_dbm = tx_power_dbm
        self.freq_hz = freq_hz
        self.sample_rate = sample_rate
        self.wavelength = 3e8 / freq_hz
        self.calibration_factor_db = None  # Será calculado automaticamente
        
        print(f"═" * 70)
        print(f"  RF Signal Simulator - PoA 1D (CALIBRAÇÃO AUTOMÁTICA)")
        print(f"═" * 70)
        print(f"  Potência TX: {tx_power_dbm} dBm | λ: {self.wavelength:.3f}m")
        print(f"═" * 70)
    
    def generate_signal(self, distance_m, duration_s=0.5, snr_db=20):
        num_samples = int(self.sample_rate * duration_s)
        t = np.arange(num_samples) / self.sample_rate
        phase = 2 * np.pi * self.freq_hz * t
        signal = np.exp(1j * phase)
        
        # Path Loss Friis
        path_loss_db = 20 * np.log10(4 * np.pi * distance_m / self.wavelength)
        rssi_theoretical_dbm = self.tx_power_dbm - path_loss_db
        
        att_linear = 10 ** (-path_loss_db / 20)
        signal_attenuated = signal * att_linear
        
        # Ruído
        noise_power = 10 ** ((rssi_theoretical_dbm - snr_db) / 10)
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return {
            'signal': signal_attenuated + noise,
            'distance_real': distance_m,
            'rssi_theoretical': rssi_theoretical_dbm,
            'path_loss_theoretical': path_loss_db,
            'num_samples': num_samples
        }
    
    def calibrate_from_known_distance(self, signal_iq, known_distance_m):
        """Calibra baseado em uma distância conhecida"""
        power_linear = np.mean(np.abs(signal_iq)**2)
        rssi_raw_db = 10 * np.log10(power_linear + 1e-12)
        
        # Path loss esperado para a distância conhecida
        path_loss_expected = 20 * np.log10(4 * np.pi * known_distance_m / self.wavelength)
        rssi_expected = self.tx_power_dbm - path_loss_expected
        
        # Fator de calibração = diferença entre esperado e medido
        self.calibration_factor_db = rssi_expected - rssi_raw_db
        print(f"  ✓ Calibração automática: {self.calibration_factor_db:+.1f} dB (distância ref: {known_distance_m}m)")
        
        return self.calibration_factor_db
    
    def calculate_rssi(self, signal_iq):
        """RSSI calibrado"""
        if self.calibration_factor_db is None:
            raise ValueError("Execute calibrate_from_known_distance() primeiro!")
        
        power_linear = np.mean(np.abs(signal_iq)**2)
        rssi_raw_db = 10 * np.log10(power_linear + 1e-12)
        rssi_calibrated = rssi_raw_db + self.calibration_factor_db
        
        return rssi_calibrated
    
    def estimate_distance(self, rssi_dbm):
        """Estima distância com modelo calibrado"""
        path_loss_db = self.tx_power_dbm - rssi_dbm
        distance = (self.wavelength / (4 * np.pi)) * 10 ** (path_loss_db / 20)
        return max(distance, 0.1)


def test_calibration_auto():
    """Teste completo com calibração automática"""
    print("\n" + "=" * 70)
    print("TESTE: CALIBRAÇÃO AUTOMÁTICA + PoA")
    print("=" * 70)
    
    simulator = RFSignalSimulator(tx_power_dbm=20)
    results = []
    
    # Distâncias de teste (inclui referência para calibração)
    test_distances = [5, 10, 15, 20, 30, 40, 50]  # 5m usado para calibração
    
    for i, distance in enumerate(test_distances):
        print(f"\nTestando {distance}m {'(CALIBRAÇÃO)' if i==0 else '(TESTE)'}")
        
        # Gerar sinal
        signal_data = simulator.generate_signal(distance, snr_db=20)
        
        if i == 0:  # Primeira medição = calibração
            simulator.calibrate_from_known_distance(signal_data['signal'], distance)
        
        # Calcular RSSI calibrado
        rssi_calibrated = simulator.calculate_rssi(signal_data['signal'])
        
        # Estimar distância
        distance_est = simulator.estimate_distance(rssi_calibrated)
        
        # Métricas
        error_m = abs(distance_est - distance)
        error_percent = (error_m / distance) * 100
        
        result = {
            'distance_real': distance,
            'distance_estimated': distance_est,
            'rssi_calibrated': rssi_calibrated,
            'rssi_theoretical': signal_data['rssi_theoretical'],
            'error_m': error_m,
            'error_percent': error_percent,
            'calibration_factor': simulator.calibration_factor_db
        }
        results.append(result)
        
        print(f"  Real:      {distance:>5.1f}m")
        print(f"  Estimado: {distance_est:>6.1f}m")
        print(f"  RSSI cal: {rssi_calibrated:>6.1f} dBm")
        print(f"  Erro:     {error_percent:>5.1f}%")
    
    # Estatísticas finais
    errors = [r['error_percent'] for r in results[1:]]  # Exclui calibração
    print(f"\n📊 RESUMO:")
    print(f"  Erro médio (pós-calibração): {np.mean(errors):.1f}%")
    print(f"  Erro máximo: {np.max(errors):.1f}%")
    print(f"  Fator calibração encontrado: {simulator.calibration_factor_db:+.1f} dB")
    
    return results


def plot_calibration_results(results):
    """Gráficos de validação"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    dist_real = [r['distance_real'] for r in results]
    dist_est = [r['distance_estimated'] for r in results]
    
    # Gráfico 1: Distância real vs estimada
    ax1.plot(dist_real, dist_est, 'ro-', linewidth=2, markersize=8)
    ax1.plot(dist_real, dist_real, 'k--', alpha=0.7, label='Linha perfeita')
    ax1.set_xlabel('Distância Real (m)')
    ax1.set_ylabel('Distância Estimada (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('PoA 1D - Calibração Automática')
    
    # Gráfico 2: Erro percentual
    errors = [r['error_percent'] for r in results[1:]]
    dist_test = [r['distance_real'] for r in results[1:]]
    ax2.plot(dist_test, errors, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Distância Real (m)')
    ax2.set_ylabel('Erro Absoluto (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Erro vs Distância')
    
    plt.tight_layout()
    plt.savefig('test_results/poa_calibrated.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráfico salvo: test_results/poa_calibrated.png")


def save_results(filename, results):
    Path('test_results').mkdir(exist_ok=True)
    full_path = f"test_results/{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ {full_path}")


def main():
    print("\n🚀 INICIANDO TESTE DE CALIBRAÇÃO AUTOMÁTICA PoA 1D")
    print("=" * 70)
    
    results = test_calibration_auto()
    
    # Salvar e plotar
    save_results("poa_calibration_auto", results)
    plot_calibration_results(results)
    
    print("\n✅ CALIBRAÇÃO CONCLUÍDA!")
    print("📁 Resultados salvos em test_results/")
    print("\nPróximo passo: GNU Radio + RTL-SDR real")


if __name__ == '__main__':
    main()
