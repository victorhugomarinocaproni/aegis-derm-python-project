import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc)
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

print(f"TensorFlow version: {tf.__version__}")

# ========================
# CONFIGURAÇÕES
# ========================
CSV_PATH = os.path.join("assets", "metadata", "metadata.csv")
IMAGES_DIR = os.path.join("assets", "images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
FINE_TUNE_AT = 30

# ========================
# 1. CARREGAR E PREPARAR DADOS
# ========================
print("\n" + "=" * 60)
print("CARREGANDO DADOS")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))
df['exists'] = df['image_path'].apply(os.path.exists)

missing = (~df['exists']).sum()
if missing > 0:
    print(f"⚠️  Aviso: {missing} imagens não encontradas")
    df = df[df['exists']].copy()

print(f"✅ Total de imagens válidas: {len(df)}")
df['label'] = df['diagnosis_1'].apply(lambda x: 'malignant' if x == 'Malignant' else 'benign')

# ========================
# 2. BALANCEAMENTO INTELIGENTE
# ========================
print("\n" + "=" * 60)
print("🔧 BALANCEANDO DATASET (ESTRATÉGIA CONSERVADORA)")
print("=" * 60)

print("ANTES do balanceamento:")
print(df['label'].value_counts())
benign_orig = (df['label'] == 'benign').sum()
malignant_orig = (df['label'] == 'malignant').sum()
print(f"Razão: {benign_orig / malignant_orig:.2f}:1\n")

df_benign = df[df['label'] == 'benign'].copy()
df_malignant = df[df['label'] == 'malignant'].copy()

# NOVA ESTRATÉGIA: Razão 2:1 (benign:malignant) SEM duplicação total
# Reduzir benignos moderadamente
target_benign = min(len(df_benign), int(len(df_malignant) * 2))

# Aumentar malignos moderadamente (apenas 50% de oversample)
target_malignant = int(len(df_malignant) * 1.5)

df_benign_sampled = resample(
    df_benign,
    n_samples=target_benign,
    random_state=42,
    replace=False
)

df_malignant_sampled = resample(
    df_malignant,
    n_samples=target_malignant,
    random_state=42,
    replace=True
)

df = pd.concat([df_benign_sampled, df_malignant_sampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("DEPOIS do balanceamento:")
print(df['label'].value_counts())
benign_new = (df['label'] == 'benign').sum()
malignant_new = (df['label'] == 'malignant').sum()
print(f"Razão: {benign_new / malignant_new:.2f}:1")
print(f"Total de imagens: {len(df)}")
print("✅ Dataset balanceado de forma conservadora!\n")

# ========================
# 3. SPLIT ESTRATIFICADO
# ========================
print("\n" + "=" * 60)
print("PREPARANDO DADOS")
print("=" * 60)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

print(f"Treino: {len(train_df)} imagens")
print(f"  - Benign: {(train_df['label'] == 'benign').sum()}")
print(f"  - Malignant: {(train_df['label'] == 'malignant').sum()}")
print(f"Validação: {len(val_df)} imagens")
print(f"  - Benign: {(val_df['label'] == 'benign').sum()}")
print(f"  - Malignant: {(val_df['label'] == 'malignant').sum()}")

# ========================
# 4. DATA AUGMENTATION MODERADO
# ========================

# Como nosso dataset é desbalanceado, nós duplicamos 50% das imangens malignas do dataset
# para balancear melhor as classes. Portanto, precisamos aplicar data augmentation moderado
# para evitar overfitting nas imagens duplicadas.
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
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['benign', 'malignant'],
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['benign', 'malignant'],
    shuffle=False,
    seed=42
)

print(f"\n✅ Generators criados!")
print(f"Classes: {train_gen.class_indices}")

# ========================
# 5. CONSTRUIR MODELO COM REGULARIZAÇÃO
# ========================
print("\n" + "=" * 60)
print("CONSTRUINDO MODELO")
print("=" * 60)

def build_model(dropout_rate=0.3, dense_units=256, l2_reg=0.001):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model()

# ========================
# 6. CLASS WEIGHTS CALCULADOS DINAMICAMENTE
# ========================
total = train_gen.n
classes_array = train_gen.classes
pos = np.sum(classes_array)  # Malignant
neg = total - pos  # Benign

# Fórmula balanceada: total / (2 * count_class)
weight_for_0 = total / (2.0 * neg)  # Benign
weight_for_1 = total / (2.0 * pos)  # Malignant

'''

-> testar sem o boost primeiro:

# Aplicar boost MODERADO para malignant (não excessivo)
boost_factor = 1.5  # Antes era implicitamente muito alto
class_weight = {
    0: weight_for_0,
    1: weight_for_1 * boost_factor
}

'''

class_weight = {
    0: weight_for_0,
    1: weight_for_1
}

print(f"\n⚖️  Class Weights Calculados:")
print(f"   Benign (0):    {class_weight[0]:.3f}")
print(f"   Malignant (1): {class_weight[1]:.3f}")
print(f"   Razão (1/0):   {class_weight[1] / class_weight[0]:.2f}x")

# print(f"   Boost aplicado: {boost_factor}x")

# ========================
# 7. COMPILAR MODELO
# ========================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

print("\n✅ Modelo construído!")
model.summary()

# ========================
# 8. CALLBACKS
# ========================
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=15,
        restore_best_weights=True,
        mode='max',
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
        'models/best_model.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# ========================
# 9. TREINAR - FASE 1
# ========================
print("\n" + "=" * 60)
print("FASE 1: TRANSFER LEARNING")
print("=" * 60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=min(EPOCHS, FINE_TUNE_AT),
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ========================
# 10. TREINAR - FASE 2 (FINE-TUNING)
# ========================
if EPOCHS > FINE_TUNE_AT:
    print("\n" + "=" * 60)
    print("FASE 2: FINE-TUNING")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=len(history1.history['loss']),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    history = {
        key: history1.history[key] + history2.history[key]
        for key in history1.history.keys()
    }
else:
    history = history1.history

# ========================
# 11. VISUALIZAR TREINAMENTO
# ========================
print("\n" + "=" * 60)
print("GERANDO VISUALIZAÇÕES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history['loss'], label='Treino', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validação', linewidth=2)
axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Época')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['accuracy'], label='Treino', linewidth=2)
axes[0, 1].plot(history['val_accuracy'], label='Validação', linewidth=2)
axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Época')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history['auc'], label='Treino', linewidth=2)
axes[1, 0].plot(history['val_auc'], label='Validação', linewidth=2)
axes[1, 0].set_title('AUC', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Época')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history['precision'], label='Precision (Treino)', linewidth=2)
axes[1, 1].plot(history['recall'], label='Recall (Treino)', linewidth=2)
axes[1, 1].plot(history['val_precision'], label='Precision (Val)', linewidth=2, linestyle='--')
axes[1, 1].plot(history['val_recall'], label='Recall (Val)', linewidth=2, linestyle='--')
axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Época')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: results/training_history.png")
plt.close()

# ========================
# 12. AVALIAR E OTIMIZAR THRESHOLD
# ========================
print("\n" + "=" * 60)
print("AVALIANDO MODELO E OTIMIZANDO THRESHOLD")
print("=" * 60)

best_model = keras.models.load_model('models/best_model.keras')

y_pred_proba = best_model.predict(val_gen, verbose=1)

y_pred_malignant = y_pred_proba[:, 1]
y_true = val_gen.classes

'''
-> TESTAR SEM ISSO

# Testar múltiplos thresholds
print("\n🎯 Testando thresholds para otimizar Recall/Precision:")
print(f"{'Thresh':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Accuracy':<10}")
print("-" * 60)

best_threshold = 0.5
best_f1 = 0
threshold_results = []

for thresh in np.arange(0.2, 0.7, 0.05):
    y_pred_test = (y_pred_proba > thresh).astype(int).flatten()

    recall = recall_score(y_true, y_pred_test)
    precision = precision_score(y_true, y_pred_test, zero_division=0)
    f1 = f1_score(y_true, y_pred_test)
    accuracy = np.mean(y_pred_test == y_true)

    threshold_results.append({
        'threshold': thresh,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy
    })

    print(f"{thresh:<10.2f} {recall:<10.2%} {precision:<12.2%} {f1:<10.3f} {accuracy:<10.2%}")

    # Critério: Recall >= 75% E melhor F1
    if recall >= 0.75 and f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n✅ Melhor threshold: {best_threshold:.2f}")
print(f"   Critério: Recall >= 75% com melhor F1-Score")
'''

y_pred = np.argmax(y_pred_proba, axis=1)

# ========================
# 13. MÉTRICAS FINAIS
# ========================
print("\n" + "=" * 60)
print("📊 MÉTRICAS FINAIS")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title(f'Matriz de Confusão ',
          fontsize=16, fontweight='bold')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Matriz de confusão salva: results/confusion_matrix.png")
plt.close()

# Análise detalhada
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n📈 Métricas Clínicas Detalhadas:")
print(f"   Sensitivity (Recall Malignant):  {sensitivity * 100:>5.1f}%  {'✅' if sensitivity >= 0.75 else '⚠️'}")
print(f"   Specificity (Recall Benign):     {specificity * 100:>5.1f}%")
print(f"   PPV (Precision Malignant):       {ppv * 100:>5.1f}%")
print(f"   NPV (Precision Benign):          {npv * 100:>5.1f}%")

print(f"\n⚠️  Casos Críticos:")
print(f"   Verdadeiros Positivos: {tp}")
print(f"   Verdadeiros Negativos: {tn}")
print(f"   Falsos Negativos (RISCO): {fn}  {'❌ CRÍTICO' if fn > tp * 0.3 else '✅'}")
print(f"   Falsos Positivos: {fp}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ ROC Curve salva: results/roc_curve.png")
plt.close()

# ========================
# 14. EXEMPLO DE USO
# ========================
print("\n" + "=" * 60)
print("EXEMPLO DE PREDIÇÃO")
print("=" * 60)

'''
=========================
ANTES COM EfficientNetB0
========================
sample_image = val_df.iloc[0]['image_path']
img = keras.preprocessing.image.load_img(sample_image, target_size=IMG_SIZE)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prob_malignant = float(best_model.predict(img_array, verbose=0)[0][1])

'''
sample_image = val_df.iloc[0]['image_path']
img = image.load_img(sample_image, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prob_malignant = float(best_model.predict(img_array, verbose=0)[0][1])
prob_benign = 1 - prob_malignant
prediction = f'Prob. de Malígno: {prob_malignant} - Prob. de Benígno: {prob_benign}'

print(f"\n📸 Imagem: {os.path.basename(sample_image)}")
print(f"Classificação: {prediction}")
print(f"Probabilidade Maligno: {prob_malignant * 100:.1f}%")
print(f"Probabilidade Benigno: {prob_benign * 100:.1f}%")

# ========================
# FINALIZAÇÃO
# ========================
print("\n" + "=" * 60)
print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 60)
print(f"""
📁 Arquivos gerados:
   • models/best_model.keras - Melhor modelo
   • results/training_history.png - Gráficos de treinamento
   • results/confusion_matrix.png - Matriz de confusão
   • results/roc_curve.png - Curva ROC

📊 Resultados Finais:
   • Sensitivity: {sensitivity * 100:.1f}%
   • Specificity: {specificity * 100:.1f}%
   • AUC: {roc_auc:.3f}

🚀 Próximos passos:
   1. Implementar Keras Tuner para otimizar hiperparâmetros
   2. Testar com dados externos (generalização)
   3. Implementar ensemble de modelos
   4. Adicionar interpretabilidade (Grad-CAM)

⚕️  LEMBRE-SE: Sistema de TRIAGEM, não diagnóstico definitivo.
   Sempre consulte um dermatologista para casos suspeitos.
""")