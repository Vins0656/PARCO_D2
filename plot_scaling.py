import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import re

def generate_plots():
    # Crea la cartella per i grafici se non esiste
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Trova tutti i file CSV nella directory corrente
    # Include sia i vecchi 'tim*.csv' che i nuovi 'weak*.csv'
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("Nessun file CSV trovato nella directory corrente.")
        return

    print(f"Trovati {len(csv_files)} file da processare: {csv_files}")

    for file_path in csv_files:
        try:
            print(f"Elaborazione file: {file_path}...")
            
            # Leggi il CSV
            df = pd.read_csv(file_path)
            
            filename_base = os.path.splitext(os.path.basename(file_path))[0]
            
            # Determina se è Weak Scaling basandosi sul nome del file
            is_weak_scaling = "weak" in filename_base.lower()

            # Verifica che le colonne necessarie esistano
            required_columns = ['Matrix', 'Procs', 'Total_Loop_Sec']
            if not all(col in df.columns for col in required_columns):
                # Silenzioso se non è un file di dati pertinente
                continue

            # Funzione per pulire i nomi delle matrici
            def clean_matrix_name(path, is_weak):
                name = os.path.basename(path).replace('.mtx', '')
                if is_weak:
                    # Rimuove il suffisso _p123 alla fine per raggruppare lo stesso kernel
                    # Esempio: weak_diag_p1 -> weak_diag
                    name = re.sub(r'_p\d+$', '', name)
                return name

            df['Matrix_Clean'] = df['Matrix'].apply(lambda x: clean_matrix_name(x, is_weak_scaling))

            # Raggruppa per Matrice e Processi e calcola la MEDIANA del tempo di loop
            grouped = df.groupby(['Matrix_Clean', 'Procs'])['Total_Loop_Sec'].median().reset_index()

            if grouped.empty:
                print(f"Nessun dato valido trovato in {file_path}")
                continue

            # Prepara i plot
            fig_main, ax_main = plt.subplots(figsize=(10, 6)) # Main plot: Speedup or Time
            fig_eff, ax_eff = plt.subplots(figsize=(10, 6))   # Efficiency plot
            
            # Lista unica delle matrici (o kernel per weak scaling)
            matrices = grouped['Matrix_Clean'].unique()
            
            # Colori per il plot
            colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))

            # Variabile per tracciare il massimo numero di processi per la linea ideale (Strong Scaling)
            global_min_procs = float('inf')
            global_max_procs = 0

            for matrix, color in zip(matrices, colors):
                # Estrai dati per la singola matrice
                mat_data = grouped[grouped['Matrix_Clean'] == matrix].sort_values('Procs')
                
                if mat_data.empty:
                    continue

                # Identifica il numero base di processi (il minimo disponibile per questa matrice)
                p_base = mat_data['Procs'].min()
                t_base = mat_data.loc[mat_data['Procs'] == p_base, 'Total_Loop_Sec'].values[0]

                # Aggiorna i limiti globali per la linea ideale
                global_min_procs = min(global_min_procs, p_base)
                global_max_procs = max(global_max_procs, mat_data['Procs'].max())

                # Calcolo metriche
                if is_weak_scaling:
                    # --- WEAK SCALING ---
                    # Plot 1: Execution Time (Idealmente costante)
                    # Plot 2: Weak Efficiency = T(base) / T(p)
                    
                    metric_main = mat_data['Total_Loop_Sec']
                    metric_eff = t_base / mat_data['Total_Loop_Sec']
                    
                    label_main = matrix
                    
                else:
                    # --- STRONG SCALING ---
                    # Plot 1: Speedup = T(base) / T(p)
                    # Plot 2: Strong Efficiency = Speedup / (p / p_base)
                    
                    speedup = t_base / mat_data['Total_Loop_Sec']
                    metric_main = speedup
                    metric_eff = speedup / (mat_data['Procs'] / p_base)
                    
                    label_main = matrix

                # Plot Main (Time or Speedup)
                ax_main.plot(mat_data['Procs'], metric_main, marker='o', label=label_main, color=color)
                
                # Plot Efficiency
                ax_eff.plot(mat_data['Procs'], metric_eff, marker='s', label=matrix, color=color)

            # --- Configurazione Plot Main ---
            if is_weak_scaling:
                ax_main.set_title(f'Weak Scaling Time - {filename_base}')
                ax_main.set_ylabel('Execution Time (s)')
                # Nota: Non tracciamo una linea ideale globale per il tempo in weak scaling
                # perché ogni kernel ha un tempo base diverso.
            else:
                ax_main.set_title(f'Strong Scaling Speedup - {filename_base}')
                ax_main.set_ylabel('Speedup (T_base / T_p)')
                
                # Linea ideale per Strong Scaling
                if global_min_procs != float('inf'):
                    ideal_x = np.array([global_min_procs, global_max_procs])
                    ideal_y = ideal_x / global_min_procs # Scaling lineare perfetto
                    ax_main.plot(ideal_x, ideal_y, 'k--', label='Ideal', alpha=0.7)

            ax_main.set_xlabel('Number of Processes')
            ax_main.grid(True, which="both", ls="-", alpha=0.5)
            if len(matrices) > 0:
                ax_main.legend()
            ax_main.set_xscale('log', base=2)
            ax_main.set_yscale('log', base=10) # Log scale anche su Y aiuta a vedere i trend
            
            # --- Configurazione Plot Efficiency ---
            ax_eff.axhline(1.0, color='k', linestyle='--', label='Ideal', alpha=0.7)
            
            title_prefix = "Weak" if is_weak_scaling else "Strong"
            ax_eff.set_title(f'{title_prefix} Parallel Efficiency - {filename_base}')
            ax_eff.set_xlabel('Number of Processes')
            ax_eff.set_ylabel('Efficiency')
            
            ax_eff.set_ylim(bottom=0) 
            
            ax_eff.grid(True, which="both", ls="-", alpha=0.5)
            if len(matrices) > 0:
                ax_eff.legend()
            ax_eff.set_xscale('log', base=2)

            # Salvataggio
            path_main = os.path.join(output_dir, f"{filename_base}_scaling.png")
            path_efficiency = os.path.join(output_dir, f"{filename_base}_efficiency.png")
            
            fig_main.savefig(path_main, dpi=300)
            fig_eff.savefig(path_efficiency, dpi=300)
            
            plt.close(fig_main)
            plt.close(fig_eff)
            print(f" -> Salvati: {path_main}, {path_efficiency}")

        except Exception as e:
            print(f"Errore processando il file {file_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_plots()