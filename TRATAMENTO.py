import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from datetime import datetime
import pytz
from functools import reduce
import time
# Define o diretório base do projeto

# Change working directory to the folder containing the files
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
start = time.time()

# Use relative paths from the current working directory
path_dados_dp = "dados_/Dados_Brutos/query (1).parquet"
path_bayer_qua = "dados_/Dados_Brutos/QUA007_2025-7-30.xls"
path_bayer_vw = "dados_/Dados_Brutos/VW_SEC002_2025-7-30.xls"
path_dados_db = "dados_/Dados_Brutos/query (2).parquet"

# Carrega os dados usando os caminhos construídos
DADOS_DP = pd.read_parquet(path_dados_dp, engine="fastparquet")
DADOS_BAYER_QUA = pd.read_excel(path_bayer_qua)
DADOS_BAYER_VW = pd.read_excel(path_bayer_vw)
DADOS_DB = pd.read_parquet(path_dados_db, engine="fastparquet")
df_bayer_qua = DADOS_BAYER_QUA.copy()
df_bayer_VW = DADOS_BAYER_VW.copy()
df_db = DADOS_DB.copy()

cols_to_split = ["Controle Secagem", "Secador.Câmara"]
for col in cols_to_split:
    df_bayer_qua[col] = df_bayer_qua[col].astype(str).str.split(";")
df_qua_expanded = df_bayer_qua.explode(cols_to_split)
df_qua_expanded = df_qua_expanded.reset_index(drop=True)

df_qua_expanded["Controle Secagem"] = df_qua_expanded["Controle Secagem"].astype(str)
df_bayer_VW["Controle Secagem"] = df_bayer_VW["Controle Secagem"].astype(str)
df_merged = pd.merge(df_qua_expanded, df_bayer_VW, on="Controle Secagem", how="left", suffixes=("_QUA", "_VW"))
df_merged.dropna(subset=["Controle Secagem", "Linha Descarga", "Início Debulha"], inplace=True)

cols_to_keep = [
    "Híbrido RNC/Linhagem_QUA",
    "Umidade (%)",
    "Início Descarga",
    "Término Descarga",
    "Linha Descarga",
    "Controle Secagem",
    "Secador.Câmara",
    "Peso Total RW  (kg)",
    "Altura camada (m)",
    "Início Secagem",
    "Inversão Câmara",
    "Término Secagem",
    "Ressecagem",
    "Secagem (horas) Atividade 2010",
    "Início Debulha",
    "Término Debulha",
    "Umidade mínima (%)",
    "Umidade máxima (%)",
    "Umidade debulha média (%)",
    "Desvio Padrão (%)",
    "Umidade Média de Fechamento AvM%",
]
df_merged = df_merged[cols_to_keep]
for col in ["Início Descarga", "Término Descarga", "Início Secagem", "Inversão Câmara", "Término Secagem", "Início Debulha", "Término Debulha"]:
    df_merged[col] = pd.to_datetime(df_merged[col], dayfirst=True)

DF_BAYER = df_merged.copy()
DF_BAYER[DF_BAYER["Linha Descarga"] == 2]

def convert_datetime_timezone(df):
    # Define the target Brazilian time zone
    brazil_timezone = pytz.timezone("America/Sao_Paulo")

    df = df[["_time", "_measurement", "_value"]]
    df = df.dropna()
    df["_time"] = df["_time"].apply(lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(brazil_timezone))
    df["_time"] = df["_time"].dt.tz_localize(None)

    return df


tags_dict = {
    "PTU_COUNT2_VC_GRAINS1/value": "COUNT_GRAINS",
    "PTU_DP2_HYB_COD/value": "HYB_COD",
    "PTU_DP2_OTM_status_atual_otimizador/value": "STATUS_OTIM",
    "PTU_DP2_RECIRCULACAO/value": "RECIRC",
    "PTU_DP2_VC_FILL2/value": "FILL",
    "PTU_DP2_VC_PERDA2/value": "PERDA",
}
df_dp = DADOS_DP.copy()
df_dp = convert_datetime_timezone(df_dp)
df_dp["_measurement"] = df_dp["_measurement"].replace(tags_dict)

dfs_dp = {}
for tag in df_dp["_measurement"].unique():
    df_tag = df_dp[df_dp["_measurement"] == tag].copy()
    dfs_dp[tag] = df_tag
def reindex_interpolating(df_tag, tag):
    df = df_tag.set_index("_time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    full_date_range = pd.date_range(start="2025-05-22 00:00:00", end="2025-07-27 23:59:59", freq="s")
    df_reindexed = df.reindex(full_date_range)

    df_reindexed["_measurement"] = df_reindexed["_measurement"].fillna(tag)
    df_reindexed["_value"] = pd.to_numeric(df_reindexed["_value"], errors="coerce")
    df_reindexed["_value"] = df_reindexed["_value"].interpolate(method="time")
    df_reindexed["_value"] = df_reindexed["_value"].ffill().bfill()  # fill head/tail gaps

    # Reset index so _time becomes a column again
    df_reindexed = df_reindexed.reset_index().rename(columns={"index": "_time"})

    return df_reindexed
dfs_dp["COUNT_GRAINS"] = reindex_interpolating(dfs_dp["COUNT_GRAINS"], "COUNT_GRAINS")
dfs_dp["FILL"] = reindex_interpolating(dfs_dp["FILL"], "FILL")
dfs_dp["PERDA"] = reindex_interpolating(dfs_dp["PERDA"], "PERDA")
dfs_dp["RECIRC"] = reindex_interpolating(dfs_dp["RECIRC"], "RECIRC")
df_com_buraco = dfs_dp["HYB_COD"].set_index("_time")
novo_indice = pd.date_range(start="2025-05-22 00:00:00", end="2025-07-27 23:59:59", freq="s")
df_reindexado = df_com_buraco.reindex(novo_indice)
df_reindexado["_measurement"] = df_reindexado["_measurement"].fillna("HYB_COD")
df_reindexado["_value"] = df_reindexado["_value"].replace(to_replace=["False", "XXXX_TESTE", "XXXXX_TESTE", "0", None], value="UNKNOWN")
dfs_dp["HYB_COD"] = df_reindexado
df_com_buraco = dfs_dp["STATUS_OTIM"].set_index("_time")
novo_indice = pd.date_range(start="2025-05-22 00:00:00", end="2025-07-27 23:59:59", freq="s")
df_reindexado = df_com_buraco.reindex(novo_indice)
df_reindexado["_measurement"] = df_reindexado["_measurement"].fillna("STATUS_OTIM")
df_reindexado["_value"] = df_reindexado["_value"].ffill()
df_reindexado["_value"] = df_reindexado["_value"].bfill()
dfs_dp["STATUS_OTIM"] = df_reindexado

tags_dict_db = {
    "PTU_DB1_HYB_COD/value": "HYB_COD",
    "PTU_DB1_BL_WT/value": "VAZAO",
    "PTU_DB1_ORIG/value": "ORIGEM",
    "PTU_DB1_NIR_UMID/value": "UMIDADE",
    "PTU_DB1_VC_PERDA/value": "PERDA",
    "PTU_DB2_HYB_COD/value": "HYB_COD",
    "PTU_DB2_BL_WT/value": "VAZAO",
    "PTU_DB2_ORIG/value": "ORIGEM",
    "PTU_DB2_NIR_UMID/value": "UMIDADE",
    "PTU_DB2_VC_PERDA/value": "PERDA",
}
df_db = convert_datetime_timezone(df_db)
df_db1 = df_db[df_db["_measurement"].str.contains("DB1")].copy()
df_db2 = df_db[df_db["_measurement"].str.contains("DB2")].copy()
df_db1["_measurement"] = df_db1["_measurement"].replace(tags_dict_db)
df_db2["_measurement"] = df_db2["_measurement"].replace(tags_dict_db)

dfs_db1 = {}
dfs_db2 = {}
for tag in df_db1["_measurement"].unique():
    df_tag = df_db1[df_db1["_measurement"] == tag].copy()
    dfs_db1[tag] = df_tag
for tag in df_db2["_measurement"].unique():
    df_tag = df_db2[df_db2["_measurement"] == tag].copy()
    dfs_db2[tag] = df_tag

for dfs_temp in [dfs_db1, dfs_db2]:
    # Fill com interpolação
    dfs_temp["VAZAO"] = reindex_interpolating(dfs_temp["VAZAO"], "VAZAO")
    dfs_temp["UMIDADE"] = reindex_interpolating(dfs_temp["UMIDADE"], "UMIDADE")
    dfs_temp["PERDA"] = reindex_interpolating(dfs_temp["PERDA"], "PERDA")

    # Fill ORIGEM com ffill e bfill
    df_com_buraco = dfs_temp["ORIGEM"].set_index("_time")
    novo_indice = pd.date_range(start="2025-05-22 00:00:00", end="2025-07-27 23:59:59", freq="s")
    df_reindexado = df_com_buraco.reindex(novo_indice)
    df_reindexado["_measurement"] = df_reindexado["_measurement"].fillna("ORIGEM")
    df_reindexado["_value"] = df_reindexado["_value"].ffill()
    df_reindexado["_value"] = df_reindexado["_value"].bfill()
    dfs_temp["ORIGEM"] = df_reindexado
for dfs_temp in [dfs_db1, dfs_db2]:
    df_com_buraco = dfs_temp["HYB_COD"].set_index("_time")
    novo_indice = pd.date_range(start="2025-05-22 00:00:00", end="2025-07-27 23:59:59", freq="s")
    df_reindexado = df_com_buraco.reindex(novo_indice)
    df_reindexado["_measurement"] = df_reindexado["_measurement"].fillna("HYB_COD")
    df_reindexado["_value"] = df_reindexado["_value"].replace(to_replace=["Sabugo", "sabugo"], value="SABUGO")
    df_reindexado["_value"] = df_reindexado["_value"].replace(to_replace=["OIU", "asgfgbngc", "False", None], value="UNKNOWN")
    dfs_temp["HYB_COD"] = df_reindexado



for var_name, dfs_dict in zip(["dfs_db1", "dfs_db2", "dfs_dp"], [dfs_db1, dfs_db2, dfs_dp]):
    dfs_renamed = []
    for name, df in dfs_dict.items():
        df = df.copy()
        if df.index.name == "_time" or "_time" not in df.columns:
            df.reset_index(inplace=True, names="_time")

        df = df[["_time", "_value"]]

        df.rename(columns={"_value": name}, inplace=True)

        dfs_renamed.append(df)

    df_merged = reduce(lambda left, right: pd.merge(left, right, on="_time", how="outer"), dfs_renamed)
    df_merged = df_merged.sort_values("_time").reset_index(drop=True)

    globals()[var_name] = df_merged

try:
    with open(r"C:\Projeto_Bayer_umidade\Dados\camaras.json", "r", encoding="utf-8") as arquivo:
        camaras = json.load(arquivo)

except FileNotFoundError:
    print("Erro: O arquivo não foi encontrado.")
except json.JSONDecodeError:
    print("Erro: O arquivo não contém um JSON válido.")

dfs_db1["ORIGEM"] = dfs_db1["ORIGEM"].map(camaras)
dfs_db2["ORIGEM"] = dfs_db2["ORIGEM"].map(camaras)

dfs_dp = dfs_dp.copy()
dfs_dp.loc[(dfs_dp["FILL"] == 0) & (dfs_dp["PERDA"] == 0), "COUNT_GRAINS"] = 0
dfs_dp["COUNT_GRAINS"] /= 10

dataframes_a_salvar = {"DF_BAYER": DF_BAYER, "DF_DP": dfs_dp, "DF_DB1": dfs_db1, "DF_DB2": dfs_db2}

os.makedirs("Dados_Consolidados", exist_ok=True)

for nome, dataframe in dataframes_a_salvar.items():
    caminho_arquivo = f"Dados_Consolidados/{nome}.pkl"
    dataframe.to_pickle(caminho_arquivo)

end = time.time()
elapsed = end - start
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Tempo gasto: {minutes:02d}:{seconds:02d} (minutos:segundos)")