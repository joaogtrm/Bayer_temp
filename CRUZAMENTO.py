import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis
import time
from scipy.stats import norm, kurtosis


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

start = time.time()

DF_BAYER = pd.read_pickle(r"Dados_Consolidados/DF_BAYER.pkl")
# print("Columns in DF_BAYER:", DF_BAYER.columns.tolist())
DF_DP = pd.read_pickle(r"Dados_Consolidados/DF_DP.pkl")
DF_DB1 = pd.read_pickle(r"Dados_Consolidados/DF_DB1.pkl")
DF_DB2 = pd.read_pickle(r"Dados_Consolidados/DF_DB2.pkl")

def delete_non_single_lines(df):
    df = df.copy()
    linha_unica = df.groupby("Controle Secagem")["Linha Descarga"].nunique() == 1
    controles_secagem_de_linha_unica = linha_unica[linha_unica].index.tolist()
    df = df[df["Controle Secagem"].isin(controles_secagem_de_linha_unica)]
    #print(df)
    return df


# Converter em timestamp
for col in ["Início Descarga", "Término Descarga", "Início Secagem", "Término Secagem", "Início Debulha", "Término Debulha"]:
    DF_BAYER[col] = pd.to_datetime(DF_BAYER[col], dayfirst=True)

# Eliminar dados referentes a câmaras enchidas por mais de uma linha de despalha
DF_BAYER = delete_non_single_lines(DF_BAYER)

# Apenas da Linha 2 onde está o Contador de Grãos
DF_BAYER = DF_BAYER[DF_BAYER["Linha Descarga"] == 2].drop_duplicates().sort_values(by="Início Descarga")

# Atribuir os valores de Controle Secagem para DP e DB's a partir das Timestamps e os dados BAYER


# Atribuir os valores de Controle Secagem para DP e DB's a partir das Timestamps e os dados BAYER
def match_timestamps(DF_BAYER, DF_TAGNA, debulha=False):
    # Garantir que as colunas de tempo sejam do tipo datetime
    DF_TAGNA["_time"] = pd.to_datetime(DF_TAGNA["_time"])
    bayer_prep = DF_BAYER.copy()

    if debulha:
        # Renomear colunas para o merge e selecionar as necessárias
        start_col, end_col = "Início Debulha", "Término Debulha"
        bayer_prep[start_col] = pd.to_datetime(bayer_prep[start_col])
        bayer_prep[end_col] = pd.to_datetime(bayer_prep[end_col])

        # Renomear a coluna de junção de câmera para corresponder ao DF_TAGNA
        bayer_prep = bayer_prep.rename(columns={start_col: "_time", "Secador.Câmara": "ORIGEM", "Híbrido RNC/Linhagem_QUA": "HYB_COD"})

        cols_to_keep = ["_time", end_col, "ORIGEM", "Controle Secagem", "HYB_COD"]
        merge_on_key = ["ORIGEM"]  # Chave de junção extra

    # Se for Despalha
    else:
        # Renomear colunas e calcular novas colunas de forma vetorizada
        start_col, end_col = "Início Descarga", "Término Descarga"
        bayer_prep[start_col] = pd.to_datetime(bayer_prep[start_col])
        bayer_prep[end_col] = pd.to_datetime(bayer_prep[end_col])

        # Cálculos vetorizados (aplicados a toda a coluna de uma vez)
        tempo_secagem_total = bayer_prep["Secagem (horas) Atividade 2010"]
        inversao_timedelta = pd.to_datetime(bayer_prep["Inversão Câmara"]) - pd.to_datetime(bayer_prep["Início Secagem"])
        bayer_prep["INVERT_PERCENTAGE"] = (inversao_timedelta.dt.total_seconds() / 3600) / tempo_secagem_total

        bayer_prep = bayer_prep.rename(
            columns={
                start_col: "_time",
                "Peso Total RW  (kg)": "BATCH_WEIGHT",
                "Altura camada (m)": "LAYER_HEIGHT",
                "Secagem (horas) Atividade 2010": "DRYING_TIME",
                "Híbrido RNC/Linhagem_QUA": "HYB_COD",
            }
        )

        cols_to_keep = ["_time", end_col, "Controle Secagem", "HYB_COD", "BATCH_WEIGHT", "LAYER_HEIGHT", "DRYING_TIME", "INVERT_PERCENTAGE", "Ressecagem"]
        merge_on_key = None  # Sem chave de junção extra

    # Manter apenas as colunas necessárias para o merge para economizar memória
    bayer_prep = bayer_prep[cols_to_keep]

    # merge_asof exige que os dataframes sejam ordenados pela chave de merge
    DF_TAGNA = DF_TAGNA.sort_values(by="_time")
    if merge_on_key:
        bayer_prep = bayer_prep.sort_values(by=["_time"] + merge_on_key)
    else:
        bayer_prep = bayer_prep.sort_values(by="_time")

    # 'direction="backward"' encontra o último valor em bayer_prep cujo '_time' é anterior ou igual ao '_time' em DF_TAGNA.
    merged_df = pd.merge_asof(DF_TAGNA, bayer_prep, on="_time", by=merge_on_key, direction="backward", suffixes=["_TAGNA", ""])

    # Após o merge, filtramos os registros onde o tempo do DF_TAGNA está fora do intervalo
    mask_intervalo = merged_df["_time"] <= merged_df[end_col]
    final_df = merged_df[mask_intervalo].copy()

    # Remove a coluna de tempo de término que não é mais necessária
    final_df = final_df.drop(columns=[end_col, "HYB_COD_TAGNA"])

    # Ordena o resultado final como na função original
    return final_df.sort_values(by=["_time", "Controle Secagem"]).reset_index(drop=True)

DP_CTRL = match_timestamps(DF_BAYER, DF_DP).drop_duplicates(subset="_time")
DP_CTRL.to_pickle("DP_CNTRL_script.pkl")

# Fillment alto faz a contagem de grãos tender a zero
def filter_despalha(df_cleaned):
    return df_cleaned.loc[
        (df_cleaned["PERDA"] > 0)
        & (df_cleaned["FILL"] < 100)
        & (df_cleaned["STATUS_OTIM"] != "LINHA DESABILITADA")
        & (df_cleaned["STATUS_OTIM"] != "AGUARDANDO MATERIAL")
        & (df_cleaned["STATUS_OTIM"] != "INICIANDO")
        & (df_cleaned["STATUS_OTIM"] != "0")
        & (df_cleaned["STATUS_OTIM"] != "SEM COMUNICAÇÃO COM AUTOMAÇÃO")
        & (df_cleaned["STATUS_OTIM"] != "SEM COMUNICAÇÃO COM VISÃO COMPUTACIONAL")
        & (df_cleaned["RECIRC"] >= 0)
        & (df_cleaned["RECIRC"] <= 800)
    ]


DP_CTRL = filter_despalha(DP_CTRL)

import matplotlib.pyplot as plt

start_date = "2025-07-02"
end_date = "2025-07-13"

fig, ax = plt.subplots()

# Plot data outside the date range
DP_CTRL_outside = DP_CTRL[(DP_CTRL["_time"] < start_date) | (DP_CTRL["_time"] > end_date)]
DP_CTRL_outside.plot(x="_time", y="COUNT_GRAINS", ax=ax, label='Período Desconsiderado')

# Plot data inside the date range in red
DP_CTRL_inside = DP_CTRL[(DP_CTRL["_time"] >= start_date) & (DP_CTRL["_time"] <= end_date)]
DP_CTRL_inside.plot(x="_time", y="COUNT_GRAINS", ax=ax, color='red', label='Período Ajustado')

ax.set_title("Período Ajustado")

# Save the figure to file
fig.savefig("time_series.png", dpi=300, bbox_inches="tight")
plt.close(fig)
DP_CTRL = DP_CTRL[(DP_CTRL["_time"] >= start_date) & (DP_CTRL["_time"] <= end_date)]



DP_series = DP_CTRL.groupby("Controle Secagem")["COUNT_GRAINS"].apply(list)

COUNT_CTRL = DP_CTRL.groupby("Controle Secagem")["COUNT_GRAINS"].sum().round()

DB1_CTRL = match_timestamps(DF_BAYER, DF_DB1, debulha=True).drop_duplicates(subset="_time")
DB1_CTRL.to_pickle("DB1_CTRL_script.pkl")


DB2_CTRL = match_timestamps(DF_BAYER, DF_DB2, debulha=True).drop_duplicates(subset="_time")

DB_series = pd.concat([DB1_CTRL.groupby("Controle Secagem")["UMIDADE"].apply(list), DB2_CTRL.groupby("Controle Secagem")["UMIDADE"].apply(list)])
#################
#################
#################
#################
#################
#################

# --- START OF CORRECTION ---

# 1. Create the base DataFrame for lots directly from DF_BAYER.
controles_validos = DP_CTRL['Controle Secagem'].unique()
df_lotes = DF_BAYER[DF_BAYER['Controle Secagem'].isin(controles_validos)].copy()

# Ensure one unique row per batch
df_lotes.drop_duplicates(subset=['Controle Secagem'], inplace=True)


# --- NEW ---
# 2. Re-calculate INVERT_PERCENTAGE as it was lost in the new logic.
#    This calculation is the primary fix.
tempo_secagem_total = df_lotes["Secagem (horas) Atividade 2010"]
inversao_timedelta = df_lotes["Inversão Câmara"] - df_lotes["Início Secagem"]
df_lotes["INVERT_PERCENTAGE"] = (inversao_timedelta.dt.total_seconds() / 3600) / tempo_secagem_total
# --- END NEW ---


# 3. Define the mapping from the original column names to the desired names.
column_mapping = {
    'Híbrido RNC/Linhagem_QUA': 'HYB_COD',
    'Peso Total RW  (kg)': 'BATCH_WEIGHT',
    'Altura camada (m)': 'LAYER_HEIGHT',
    'Secagem (horas) Atividade 2010': 'DRYING_TIME',
    'Ressecagem': 'Ressecagem'
}

# 4. Select and rename the columns.
#    We now include 'INVERT_PERCENTAGE' in the list of columns to process.
columns_to_process = ['Controle Secagem', 'INVERT_PERCENTAGE'] + list(column_mapping.keys())
df_lotes = df_lotes[columns_to_process].rename(columns=column_mapping)


# 5. Set 'Controle Secagem' as the index...
df_lotes.set_index('Controle Secagem', inplace=True)


# 6. Join the aggregated time-series data...
df_lotes["COUNT_GRAINS"] = DP_series
df_lotes = df_lotes.join(DB_series.rename("UMIDADE"), how="left").dropna(subset=['UMIDADE', 'COUNT_GRAINS'])


# 7. Now, create the final DataFrame for analysis.
# --- MODIFIED ---
#    Update the list of columns to keep to include the new INVERT_PERCENTAGE.
cols_to_keep_final = ['HYB_COD', 'BATCH_WEIGHT', 'LAYER_HEIGHT', 'DRYING_TIME', 'INVERT_PERCENTAGE', 'Ressecagem']
# --- END MODIFIED ---
df_lotes_final = df_lotes[cols_to_keep_final].copy()
counts_df = pd.DataFrame(df_lotes["COUNT_GRAINS"].tolist(), index=df_lotes.index)
df_lotes_final["media_COUNT_GRAINS"] = counts_df.mean(axis=1)
df_lotes_final["std_dev_COUNT_GRAINS"] = counts_df.std(axis=1)
df_lotes_final["max_COUNT_GRAINS"] = counts_df.max(axis=1)
df_lotes_final["soma_COUNT_GRAINS"] = counts_df.sum(axis=1)

umidade_df = pd.DataFrame(df_lotes["UMIDADE"].tolist(), index=df_lotes.index)
Q1 = umidade_df.quantile(0.25, axis=1)
Q3 = umidade_df.quantile(0.75, axis=1)
df_lotes_final["umidade_IQR"] = Q3 - Q1

# Calcular a matriz de correlação (Spearman é recomendado)
matriz_corr = df_lotes_final.select_dtypes(include=["number"]).corr(method="spearman")

# Visualizar a matriz de correlação com um heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação de Spearman entre Atributos de Lote")
plt.savefig(r"plots\heatmap_correlacao.png", dpi=300, bbox_inches="tight")
plt.close()


# (Opcional) Se você precisasse das outras métricas de umidade, elas também seriam vetorizadas:
# n_observacoes = 30
# df_lotes_final["umidade_inicial_media"] = umidade_df.iloc[:, :n_observacoes].mean(axis=1)
# df_lotes_final["umidade_final_media"] = umidade_df.iloc[:, -n_observacoes:].mean(axis=1)
# df_lotes_final["delta_umidade"] = df_lotes_final["umidade_final_media"] - df_lotes_final["umidade_inicial_media"]
numeric_cols = ["BATCH_WEIGHT", "LAYER_HEIGHT", "DRYING_TIME", "INVERT_PERCENTAGE"]

def generate_correlation_plots_subplots(df, column_to_use):
    # q=3 especifica que queremos tercis (0-33%, 33-66%, 66-100%).
    # retbins=True retorna os pontos de quebra dos tercis.
    df["Nivel_Contaminacao"], break_points = pd.qcut(df[column_to_use], q=3, labels=["Baixo", "Médio", "Alto"], retbins=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 24))

    fig.suptitle(f"Matriz de Correlação de Spearman por Nível de '{column_to_use}'", fontsize=18, y=1.02)

    labels = ["Baixo", "Médio", "Alto"]

    for i, label in enumerate(labels):
        filtered_df = df[df["Nivel_Contaminacao"] == label]
        numeric_df = filtered_df.select_dtypes(include=["number"]).drop(numeric_cols + ["std_dev_COUNT_GRAINS"], axis=1)
        matriz_corr = numeric_df.corr(method="spearman")
        sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[i])
        axes[i].set_title(f"Nível {label} (Entre {break_points[i]:.2f} e {break_points[i + 1]:.2f})")

    plt.tight_layout()
    plt.close()


for num_col in numeric_cols:
    generate_correlation_plots_subplots(df_lotes_final, num_col)


import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def gerar_grid_correlacao_separada(
    dataframe: pd.DataFrame,
    analysis_cols: list,
    q: int = 3,
    labels: list = ["Baixo", "Médio", "Alto"],
    base_title: str = "Matriz de Correlação de Spearman por Nível de Variáveis",
    cols_to_drop_from_corr: list = ["BATCH_WEIGHT", "LAYER_HEIGHT", "DRYING_TIME", "INVERT_PERCENTAGE"],
    save_folder: str = "plots_cor"
):
    if len(labels) != q:
        raise ValueError(f"O número de 'labels' ({len(labels)}) deve ser igual ao valor de 'q' ({q}).")
    
    # Create folder to save plots if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    
    for column_to_use in analysis_cols:
        df_temp = dataframe.copy()
        
        try:
            df_temp["Nivel_Analise"], break_points = pd.qcut(df_temp[column_to_use], q=q, labels=labels, retbins=True, duplicates="drop")
        except ValueError:
            print(f"Aviso: Não foi possível criar {q} quantis para a coluna '{column_to_use}'. Pulando esta coluna.")
            continue
        
        for row_idx, label in enumerate(labels):
            filtered_df = df_temp[df_temp["Nivel_Analise"] == label]
            
            # Columns to drop from correlation calculation
            cols_to_drop = analysis_cols + cols_to_drop_from_corr
            numeric_df = filtered_df.select_dtypes(include=["number"]).drop(columns=cols_to_drop, errors="ignore")
            
            if numeric_df.empty:
                print(f"Warning: No numeric data remaining after dropping columns for '{column_to_use}' nivel '{label}'. Skipping.")
                continue

            matriz_corr = numeric_df.corr(method="spearman")
            
            # Create figure for each individual correlation plot
            plt.figure(figsize=(8, 6))
            
            sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10}, yticklabels=True)
            
            title_line1 = f"Análise por: '{column_to_use}'"
            title_line2 = f"Nível {label} ({break_points[row_idx]:.2f} a {break_points[row_idx + 1]:.2f})"
            plt.title(f"{base_title}\n{title_line1}\n{title_line2}", fontsize=16)
            
            plt.tight_layout()
            
            # Save plot to file naming after column and level
            safe_label = str(label).replace(" ", "_").replace("/", "_")
            filename = f"{save_folder}/correlacao_{column_to_use}_{safe_label}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

            print(f"Plot saved: {filename}")

gerar_grid_correlacao_separada(
    df_lotes_final,
    analysis_cols=["BATCH_WEIGHT", "LAYER_HEIGHT", "DRYING_TIME", "INVERT_PERCENTAGE"]
)

gerar_grid_correlacao_separada(
    dataframe=df_lotes_final,
    analysis_cols=["BATCH_WEIGHT", "DRYING_TIME"],
    q=2,
    labels=["Abaixo da Mediana", "Acima da Mediana"],
    base_title="Análise de Correlação (Abaixo vs. Acima da Mediana)"
)



def generate_box_plots(df, column_to_use, save_path=None):
    # q=3 especifica tercis (0-33%, 33-66%, 66-100%)
    if save_path:
        base = os.path.basename(save_path)
        if base.startswith('boxplot_'):
            parts = base.split('_')
            if len(parts) > 1:
                title_suffix = parts[1]  # palavra após 'boxplot_'
    df["Nivel_Contaminacao"] = pd.qcut(df[column_to_use], q=3, labels=["Baixo", "Médio", "Alto"])
    contagem_lotes = df["Nivel_Contaminacao"].value_counts()
    subtitulo = f"Contagem de Lotes: Baixo ({contagem_lotes.get('Baixo', 0)}), Médio ({contagem_lotes.get('Médio', 0)}), Alto ({contagem_lotes.get('Alto', 0)})"

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Nivel_Contaminacao",
        y="umidade_IQR",
        data=df,
        order=["Baixo", "Médio", "Alto"],
    )
    if title_suffix == 'soma':
        title_main = "Soma da Contagem de Grãos por Nível de Contaminação"
    elif title_suffix == 'max':
        title_main = "Máximo da Contagem de Grãos por Nível de Contaminação"
    elif title_suffix == 'media':
        title_main = "Média da Contagem de Grãos por Nível de Contaminação"
    else:
        title_main = f"Dispersão da Umidade (IQR) por Nível de Contaminação de Grãos - {title_suffix if title_suffix else column_to_use}"

    plt.suptitle(title_main, fontsize=16, y=0.97)
    plt.title(subtitulo, fontsize=12)
    plt.xlabel(f"Nível de {column_to_use}", fontsize=12)
    plt.ylabel("Variabilidade da Umidade (IQR)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Fecha a figura para evitar sobreposição
    else:
        plt.close()

generate_box_plots(df_lotes_final, "max_COUNT_GRAINS", r"plots\boxplot_max_COUNT_GRAINS.png")
generate_box_plots(df_lotes_final, "media_COUNT_GRAINS", r"plots\boxplot_media_COUNT_GRAINS.png")
generate_box_plots(df_lotes_final, "soma_COUNT_GRAINS", r"plots\boxplot_soma_COUNT_GRAINS.png")



# Set a style for better-looking plots
sns.set_style("whitegrid")

from scipy.stats import norm, kurtosis

# Set a style for better-looking plots
sns.set_style("whitegrid")


def cap_outliers(df, column):
    # Identify outlier boundaries using IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_capped = df.copy()

    # "Cap" the outlier values at the lower and upper bounds
    df_capped[column] = np.clip(df_capped[column], a_min=lower_bound, a_max=upper_bound)

    return df_capped


def plot_distributions(df, column, df_name, bins=100):
    # Cap dos outliers
    df_capped = cap_outliers(df, column)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Análise do DataFrame {df_name}", fontsize=16)

    # Distribuição Real (KDE)
    sns.histplot(data=df_capped, x=column, kde=True, ax=axes[0], bins=bins)
    axes[0].set_title(f"Distribuição Real de {column}")
    axes[0].set_xlabel(f"{column}")
    axes[0].set_ylabel("Frequência")

    # Estima os parâmetros (média e desvio padrão)
    mu, sigma = norm.fit(df_capped[column])
    data_kurt = kurtosis(df_capped[column])

    # Plota o histograma dos dados
    sns.histplot(data=df_capped, x=column, bins=bins, stat="density", color="lightblue", label="Histograma dos Dados", ax=axes[1])

    # Cria a linha da distribuição normal ajustada
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    axes[1].plot(x, p, "k", linewidth=2, label="Distribuição Normal Ajustada")

    # Parâmetros da figura
    axes[1].set_title(f"Distribuição Normal Ajustada de {column}\n(Média={mu:.2f}, Desvio Padrão={sigma:.2f}, Kurtosis={data_kurt:.2f})")
    axes[1].set_xlabel(column)
    axes[1].set_ylabel("Densidade")
    axes[1].legend()

    plt.tight_layout()

def fit_batch_data(data_df, ids):
    individual_params = []
    for batch_id in ids:
        # Filter data for the current ID
        data_subset = data_df[data_df["Controle Secagem"] == batch_id]["UMIDADE"]

        # Ensure there is enough data to fit
        if len(data_subset) < 2:
            print(f"⚠️ Warning: Not enough data for ID '{batch_id}'. Skipping.")
            continue

        # Fit a normal distribution to the subset
        mu, sigma = norm.fit(data_subset)

        # Store the parameters (mean and standard deviation)
        individual_params.append({"id": batch_id, "mu": mu, "sigma": sigma})

        print(f"ID '{batch_id}': Mean = {mu:.4f}, Std Dev = {sigma:.4f}")

    # Extract all means and variances (sigma^2)
    mus = [p["mu"] for p in individual_params]
    variances = [p["sigma"] ** 2 for p in individual_params]
    n = len(individual_params)

    # Calculate the parameters of the final averaged distribution
    if n > 0:
        mu_final = np.mean(mus)
        variance_final = np.sum(variances) / (n**2)
        sigma_final = np.sqrt(variance_final)

        print(f"Number of distributions averaged: {n}")
        print("\n✅ Final Averaged Distribution:")
        print(f"   Final Mean = {mu_final:.4f}")
        print(f"   Final Std Dev = {sigma_final:.4f}")
        print("####################################\n")
    else:
        print("No distributions were fitted, cannot compute an average.")


def analyse_hybrid(DP_CTRL, DF_DB_CTRL, HYB_COD):
    # Medida agregada de COUNT_GRAINS para cada Controle Secagem (estimativa da quantidade de grãos dentro de cada camara)
    COUNT_CTRL = DP_CTRL.groupby("Controle Secagem")["COUNT_GRAINS"].sum().round()

    # Debulha filtrada por híbrido
    DB_CTRL_HYB = DF_DB_CTRL[DF_DB_CTRL["HYB_COD"] == HYB_COD]

    # Medidas de COUNT_GRAINS do híbrido
    COUNT_CTRL_HYB = COUNT_CTRL[COUNT_CTRL.index.isin(DB_CTRL_HYB["Controle Secagem"].unique())]

    # Plot teste com o Controle Secagem com a menor e a maior contagem de grãos
    if len(COUNT_CTRL_HYB) != 0:
        plot_distributions(DB_CTRL_HYB[DB_CTRL_HYB["Controle Secagem"] == COUNT_CTRL_HYB.idxmin()], "UMIDADE", f"{HYB_COD}_MIN", bins=30)
        plot_distributions(DB_CTRL_HYB[DB_CTRL_HYB["Controle Secagem"] == COUNT_CTRL_HYB.idxmax()], "UMIDADE", f"{HYB_COD}_MAX", bins=30)

    # Separar em dois grupos de distribuições menor e maior que mediana de COUNT_GRAINS para verificar um comportamento mais global
    CTRL_HIGH_HYB = COUNT_CTRL_HYB.loc[COUNT_CTRL_HYB >= COUNT_CTRL_HYB.median()].index.to_list()
    CTRL_LOW_HYB = COUNT_CTRL_HYB.loc[COUNT_CTRL_HYB < COUNT_CTRL_HYB.median()].index.to_list()

    # Fit de uma normal para cada Controle Secagem e cálculo da distribuição global
    print(f"GRUPO HIGH_COUNT {HYB_COD}\n")
    fit_batch_data(DB_CTRL_HYB, CTRL_HIGH_HYB)
    print(f"GRUPO LOW_COUNT {HYB_COD}\n")
    fit_batch_data(DB_CTRL_HYB, CTRL_LOW_HYB)

#for hyb in DB1_CTRL["HYB_COD"].unique():
#    analyse_hybrid(DP_CTRL, DB1_CTRL, hyb)







end = time.time()
elapsed = end - start
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Tempo gasto: {minutes:02d}:{seconds:02d} (minutos:segundos)")