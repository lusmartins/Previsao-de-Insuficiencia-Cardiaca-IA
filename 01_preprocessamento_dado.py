import pandas as pd
core= pd.read_csv('dataset/heart_desease.csv')

print(core.head())

# Verificação de celulas vazias ou Nan.
print("\nValores ausentes por coluna:")
print(core.isna().sum())

# Criação de um novo dataframe caso existam valores ausentes
if core.isna().sum().sum() > 0:
    core_limpo = core.dropna().copy()
    print("\nValores ausentes encontrados e removidos.")
else:
    core_limpo = core.copy()
    print("\nNenhum valor ausente encontrado.")



#distribuição de classes que devem ser classificadas
colunas = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease"
]
core_final = core_limpo[colunas].copy()


#dataframe final e distribuição das classes
print("\nDataFrame Final:")
print(core_final.head())

print("\nDistribuição da variável alvo:")
print(core_final[["HeartDisease"]].value_counts())

#conversão de classes para atributos numéricos
core_final["Sex"] = core_final["Sex"].map({
    "M": 0,
    "F": 1
})

core_final["ChestPainType"] = core_final["ChestPainType"].map({
    "TA": 0,
    "ATA": 1,
    "NAP": 2,
    "ASY": 3
})

core_final["RestingECG"] = core_final["RestingECG"].map({
    "Normal": 0,
    "ST": 1,
    "LVH": 2
})

core_final["ExerciseAngina"] = core_final["ExerciseAngina"].map({
    "N": 0,
    "Y": 1
})

core_final["ST_Slope"] = core_final["ST_Slope"].map({
    "Up": 0,
    "Flat": 1,
    "Down": 2
})

print(core_final.head())


#dataset atualizado 
#heart_desease_ajustado = core
hda=core_final
print(hda)
hda.to_csv('dataset/heart_desease_ajustado.csv', index=False)

