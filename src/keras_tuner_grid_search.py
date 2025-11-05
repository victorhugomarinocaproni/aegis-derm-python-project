"""
Script de Otimização de Hiperparâmetros com Keras Tuner
Para classificação de lesões de pele (Benign vs Malignant)

IMPORTANTE: Instale antes:
pip install keras-tuner
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Keras Tuner
import keras_tuner as kt

# ========================
# CONFIGURAÇÕES
# ========================
CSV_PATH = os.path.join("assets", "metadata", "metadata.csv")
IMAGES_DIR = os.path.join("assets", "images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_TRIALS = 20  # Número de combinações a testar
EXECUTIONS_PER_TRIAL = 1  # Número de vezes que cada combinação é treinada

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")

# Configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU configurada")
    except RuntimeError as e:
        print(f"⚠️  Erro ao configurar GPU: {e}")

# ========================
# PREPARAR DADOS (MESMO CÓDIGO)
# ========================
print("\n" + "=" * 60)
print("PREPARANDO DADOS")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))
df['exists'] = df['image_path'].apply(os.path.exists)
df = df[df['exists']].copy()
df['label'] = df['diagnosis_1'].apply(lambda x: 'malignant' if x == 'Malignant' else 'benign')

# Balanceamento
df_benign = df[df['label'] == 'benign'].copy()
df_malignant = df[df['label'] == 'malignant'].copy()

target_benign = min(len(df_benign), int(len(df_malignant) * 2))
target_malignant = int(len(df_malignant) * 1.5)

df_benign_sampled = resample(df_benign, n_samples=target_benign, random_state=42, replace=False)
df_malignant_sampled = resample(df_malignant, n_samples=target_malignant, random_state=42, replace=True)

df = pd.concat([df_benign_sampled, df_malignant_sampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset balanceado: {len(df)} imagens")
print(df['label'].value_counts())

# Split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

print(f"Treino: {len(train_df)} | Validação: {len(val_df)}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', classes=['benign', 'malignant'],
    shuffle=True, seed=42
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', classes=['benign', 'malignant'],
    shuffle=False, seed=42
)

# Class weights
total = train_gen.n
pos = np.sum(train_gen.classes)
neg = total - pos

weight_for_0 = total / (2.0 * neg)
weight_for_1 = total / (2.0 * pos)


# ========================
# DEFINIR MODELO PARA TUNING
# ========================
def build_model_for_tuning(hp):
    """
    Função que constrói modelos com hiperparâmetros variáveis
    hp: HyperParameters object do Keras Tuner
    """

    # Base model (fixo)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    # Hiperparâmetros para tuning
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    dense_units_1 = hp.Choice('dense_units_1', values=[128, 256, 512])
    dense_units_2 = hp.Choice('dense_units_2', values=[64, 128, 256])
    l2_reg = hp.Float('l2_regularization', min_value=1e-4, max_value=1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    boost_factor = hp.Float('boost_factor', min_value=1.0, max_value=3.0, step=0.5)

    # Construir modelo
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        dense_units_1,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        dense_units_2,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    # Class weights dinâmicos
    class_weight = {
        0: weight_for_0,
        1: weight_for_1 * boost_factor
    }

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision')
        ]
    )

    # Salvar class_weight no modelo para usar no fit
    model.class_weight = class_weight

    return model


# ========================
# CONFIGURAR KERAS TUNER
# ========================
print("\n" + "=" * 60)
print("CONFIGURANDO KERAS TUNER")
print("=" * 60)

# Criar diretório para resultados do tuner
os.makedirs('tuner_results', exist_ok=True)

# Usar RandomSearch (mais rápido) ou BayesianOptimization (mais inteligente)
tuner = kt.BayesianOptimization(
    build_model_for_tuning,
    objective=kt.Objective('val_recall', direction='max'),  # Maximizar Recall
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory='tuner_results',
    project_name='skin_lesion_tuning',
    overwrite=False  # True para recomeçar do zero
)

print("✅ Tuner configurado!")
print(f"   Objetivo: Maximizar Recall de validação")
print(f"   Max trials: {MAX_TRIALS}")
print(f"   Executions per trial: {EXECUTIONS_PER_TRIAL}")

# Visualizar espaço de busca
print("\n📊 Espaço de busca:")
tuner.search_space_summary()

# ========================
# EXECUTAR BUSCA
# ========================
print("\n" + "=" * 60)
print("INICIANDO BUSCA DE HIPERPARÂMETROS")
print("=" * 60)
print("⏳ Isso pode demorar várias horas...")

# Callback para early stopping durante busca
early_stop = EarlyStopping(
    monitor='val_recall',
    patience=10,
    mode='max',
    restore_best_weights=True,
    verbose=0
)


# Custom callback para exibir progresso
class ProgressCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"   Epoch {epoch + 1}: Recall={logs['val_recall']:.3f}, AUC={logs['val_auc']:.3f}")


# Executar busca
tuner.search(
    train_gen,
    validation_data=val_gen,
    epochs=30,  # Reduzido para busca mais rápida
    callbacks=[early_stop],
    verbose=0
)

print("\n✅ Busca concluída!")

# ========================
# ANALISAR RESULTADOS
# ========================
print("\n" + "=" * 60)
print("📊 MELHORES CONFIGURAÇÕES")
print("=" * 60)

# Obter melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=3)

for i, hp in enumerate(best_hps, 1):
    print(f"\n🏆 Configuração #{i}:")
    print(f"   Dropout rate: {hp.get('dropout_rate'):.2f}")
    print(f"   Dense units 1: {hp.get('dense_units_1')}")
    print(f"   Dense units 2: {hp.get('dense_units_2')}")
    print(f"   L2 regularization: {hp.get('l2_regularization'):.6f}")
    print(f"   Learning rate: {hp.get('learning_rate'):.6f}")
    print(f"   Boost factor: {hp.get('boost_factor'):.2f}")

# Obter melhor modelo
print("\n" + "=" * 60)
print("TREINANDO MELHOR MODELO COMPLETO")
print("=" * 60)

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Retreinar com mais épocas
print("\n⚙️  Retreinando melhor modelo com 50 épocas...")

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

callbacks_final = [
    EarlyStopping(
        monitor='val_recall',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_tuned_model.keras',
        monitor='val_recall',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

history = best_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks_final,
    class_weight=best_model.class_weight,
    verbose=1
)

print("\n✅ Treinamento final concluído!")

# ========================
# AVALIAR MODELO FINAL
# ========================
print("\n" + "=" * 60)
print("AVALIANDO MODELO OTIMIZADO")
print("=" * 60)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

best_model_loaded = keras.models.load_model('models/best_tuned_model.keras')

y_pred_proba = best_model_loaded.predict(val_gen, verbose=1)
y_true = val_gen.classes

# Encontrar melhor threshold
from sklearn.metrics import recall_score, precision_score, f1_score

best_threshold = 0.5
best_f1 = 0

print("\n🎯 Otimizando threshold:")
for thresh in np.arange(0.2, 0.7, 0.05):
    y_pred_test = (y_pred_proba > thresh).astype(int).flatten()
    recall = recall_score(y_true, y_pred_test)
    precision = precision_score(y_true, y_pred_test, zero_division=0)
    f1 = f1_score(y_true, y_pred_test)

    print(f"   Thresh={thresh:.2f}: Recall={recall:.2%}, Precision={precision:.2%}, F1={f1:.3f}")

    if recall >= 0.75 and f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n✅ Threshold ótimo: {best_threshold:.2f}")

y_pred = (y_pred_proba > best_threshold).astype(int).flatten()

# Métricas finais
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title(f'Matriz de Confusão - Modelo Otimizado\n(Threshold = {best_threshold:.2f})',
          fontsize=14, fontweight='bold')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix_tuned.png', dpi=300, bbox_inches='tight')
print("✅ Matriz salva: results/confusion_matrix_tuned.png")
plt.close()

# Métricas clínicas
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n📈 Métricas Clínicas:")
print(f"   Sensitivity: {sensitivity * 100:.1f}%  {'✅' if sensitivity >= 0.75 else '⚠️'}")
print(f"   Specificity: {specificity * 100:.1f}%")
print(f"   PPV: {ppv * 100:.1f}%")
print(f"   NPV: {npv * 100:.1f}%")
print(f"\n   Falsos Negativos: {fn}  {'❌ CRÍTICO' if fn > tp * 0.3 else '✅'}")
print(f"   Falsos Positivos: {fp}")

# ========================
# SALVAR CONFIGURAÇÃO FINAL
# ========================
import json

config = {
    'best_hyperparameters': {
        'dropout_rate': float(best_hp.get('dropout_rate')),
        'dense_units_1': int(best_hp.get('dense_units_1')),
        'dense_units_2': int(best_hp.get('dense_units_2')),
        'l2_regularization': float(best_hp.get('l2_regularization')),
        'learning_rate': float(best_hp.get('learning_rate')),
        'boost_factor': float(best_hp.get('boost_factor'))
    },
    'best_threshold': float(best_threshold),
    'metrics': {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'false_negatives': int(fn),
        'false_positives': int(fp)
    }
}

with open('models/best_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n✅ Configuração salva em: models/best_config.json")

print("\n" + "=" * 60)
print("✅ OTIMIZAÇÃO CONCLUÍDA!")
print("=" * 60)
print(f"""
📁 Arquivos gerados:
   • models/best_tuned_model.keras - Modelo otimizado
   • models/best_config.json - Configuração ótima
   • results/confusion_matrix_tuned.png - Matriz de confusão
   • tuner_results/ - Logs do Keras Tuner

🎯 Melhores hiperparâmetros encontrados!
   Use o arquivo best_config.json para reproduzir os resultados.
""")