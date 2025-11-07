import json
import logging
import os
from datetime import datetime
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras_tuner as kt


# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

class LoggerConfig:
    """Configuração centralizada de logging."""

    @staticmethod
    def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
        """Configura e retorna um logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


# ============================================================================
# CONFIGURAÇÕES DO PROJETO
# ============================================================================

class Config:
    """Classe de configuração centralizada."""

    CSV_PATH = os.path.join("assets", "metadata", "metadata.csv")
    IMAGES_DIR = os.path.join("assets", "images")
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    EPOCHS = 50
    FINE_TUNE_AT = 30
    FINE_TUNE_LAYERS = 30

    N_FOLDS = 5

    MAX_TRIALS = 20
    EXECUTIONS_PER_TRIAL = 2
    TUNER_DIR = "keras_tuner"
    TUNER_PROJECT_NAME = "skin_cancer_resnet50"

    RANDOM_SEED = 42

    @classmethod
    def create_directories(cls):
        """Cria todos os diretórios necessários."""
        for directory in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR, cls.TUNER_DIR]:
            os.makedirs(directory, exist_ok=True)


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Gerencia carregamento e preparação dos dados."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Carrega e prepara o dataset."""
        self.logger.info("=" * 60)
        self.logger.info("CARREGANDO E PREPARANDO DADOS")
        self.logger.info("=" * 60)

        df = pd.read_csv(self.config.CSV_PATH)
        df['image_path'] = df['isic_id'].apply(
            lambda x: os.path.join(self.config.IMAGES_DIR, f"{x}.jpg")
        )
        df['exists'] = df['image_path'].apply(os.path.exists)

        missing = (~df['exists']).sum()
        if missing > 0:
            self.logger.warning(f"⚠  {missing} imagens não encontradas")
            df = df[df['exists']].copy()

        self.logger.info(f" Total de imagens válidas: {len(df)}")

        df['label'] = df['diagnosis_1'].apply(
            lambda x: 'malignant' if x == 'Malignant' else 'benign'
        )

        return df

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica balanceamento conservador ao dataset."""
        self.logger.info("=" * 60)
        self.logger.info("BALANCEANDO DATASET")
        self.logger.info("=" * 60)

        benign_count = (df['label'] == 'benign').sum()
        malignant_count = (df['label'] == 'malignant').sum()

        self.logger.info(f"ANTES - Benign: {benign_count}, Malignant: {malignant_count}")
        self.logger.info(f"Razão: {benign_count / malignant_count:.2f}:1")

        df_benign = df[df['label'] == 'benign'].copy()
        df_malignant = df[df['label'] == 'malignant'].copy()

        # Estratégia: Razão 2:1 (benign:malignant)
        target_benign = min(len(df_benign), int(len(df_malignant) * 2))
        target_malignant = int(len(df_malignant) * 1.5)

        df_benign_sampled = resample(
            df_benign,
            n_samples=target_benign,
            random_state=self.config.RANDOM_SEED,
            replace=False
        )

        df_malignant_sampled = resample(
            df_malignant,
            n_samples=target_malignant,
            random_state=self.config.RANDOM_SEED,
            replace=True
        )

        df_balanced = pd.concat([df_benign_sampled, df_malignant_sampled])
        df_balanced = df_balanced.sample(
            frac=1,
            random_state=self.config.RANDOM_SEED
        ).reset_index(drop=True)

        benign_new = (df_balanced['label'] == 'benign').sum()
        malignant_new = (df_balanced['label'] == 'malignant').sum()

        self.logger.info(f"DEPOIS - Benign: {benign_new}, Malignant: {malignant_new}")
        self.logger.info(f"Razão: {benign_new / malignant_new:.2f}:1")
        self.logger.info(f"Total de imagens: {len(df_balanced)}")

        return df_balanced

    def create_data_generators(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Cria os geradores de dados com data augmentation."""

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='label',
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            classes=['benign', 'malignant'], # --> Convenção: a classe 1 (em classificações binárias) é sempre a classe de interesse
            shuffle=True,                    # e será vista como "POSITIVE" na matriz de confusão e métricas derivadas.g
            seed=self.config.RANDOM_SEED
        )

        val_gen = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='label',
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            classes=['benign', 'malignant'], # --> Convenção: a classe 1 (em classificações binárias) é sempre a classe de interesse
            shuffle=False,                   # e será vista como "POSITIVE" na matriz de confusão e métricas derivadas.
            seed=self.config.RANDOM_SEED
        )

        return train_gen, val_gen


# ============================================================================
# MODEL BUILDER
# ============================================================================

class ModelBuilder:
    """Constrói modelos com diferentes configurações."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def build_model(
            self,
            dropout_rate: float = 0.3,
            dense_units: int = 256,
            l2_reg: float = 0.001,
            learning_rate: float = 1e-3
    ) -> Tuple[keras.Model, keras.Model]:
        """Constrói o modelo ResNet50 com camadas customizadas."""

        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(*self.config.IMG_SIZE, 3))
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
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        return model, base_model

    def calculate_class_weights(self, train_gen: ImageDataGenerator) -> Dict[int, float]:
        """Calcula os pesos das classes de forma balanceada."""
        total = train_gen.n
        classes_array = train_gen.classes
        pos = np.sum(classes_array)
        neg = total - pos

        weight_for_0 = total / (2.0 * neg)
        weight_for_1 = total / (2.0 * pos)

        class_weight = {
            0: weight_for_0,
            1: weight_for_1
        }

        self.logger.info(f"  Class Weights:")
        self.logger.info(f"   Benign (0):    {class_weight[0]:.3f}")
        self.logger.info(f"   Malignant (1): {class_weight[1]:.3f}")
        self.logger.info(f"   Razão (1/0):   {class_weight[1] / class_weight[0]:.2f}x")

        return class_weight


# ============================================================================
# KERAS TUNER INTEGRATION
# ============================================================================

class ResNet50HyperModel(kt.HyperModel):
    """HyperModel para busca de hiperparâmetros com Keras Tuner."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """Constrói o modelo com hiperparâmetros."""

        # Hiperparâmetros a serem otimizados
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        dense_units = hp.Choice('dense_units', values=[128, 256, 512])
        l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

        # Construir modelo
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(*self.config.IMG_SIZE, 3))
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
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        return model


class HyperparameterTuner:
    """Gerencia a busca de hiperparâmetros com Keras Tuner."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.best_hps = None

    def search_hyperparameters(
            self,
            train_gen: ImageDataGenerator,
            val_gen: ImageDataGenerator,
            class_weight: Dict[int, float],
            search_hyperparameters: bool
    ) -> kt.HyperParameters:
        """Executa a busca de hiperparâmetros."""

        self.logger.info("=" * 60)
        self.logger.info("INICIANDO BUSCA DE HIPERPARÂMETROS COM KERAS TUNER")
        self.logger.info("=" * 60)

        hypermodel = ResNet50HyperModel(self.config)

        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=kt.Objective('val_auc', direction='max'), # -> talvez mexer nesse objetivo (mudar para recall) ??
            max_trials=self.config.MAX_TRIALS,
            executions_per_trial=self.config.EXECUTIONS_PER_TRIAL,
            directory=self.config.TUNER_DIR,
            project_name=self.config.TUNER_PROJECT_NAME,
            overwrite=False
        )

        self.logger.info(f"Configuração do Tuner:")
        self.logger.info(f"  - Algoritmo: Bayesian Optimization")
        self.logger.info(f"  - Max Trials: {self.config.MAX_TRIALS}")
        self.logger.info(f"  - Executions per Trial: {self.config.EXECUTIONS_PER_TRIAL}")
        self.logger.info(f"  - Objetivo: Maximizar val_auc")

        callbacks = [
            EarlyStopping(
                monitor='val_auc', # -> mudar para val_recall ??
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]

        class TunerLogger(keras.callbacks.Callback):
            def __init__(self, logger, trial_num):
                super().__init__()
                self.logger = logger
                self.trial_num = trial_num

            def on_epoch_end(self, epoch, logs=None):
                self.logger.info(
                    f"Trial {self.trial_num} - Época {epoch + 1} - "
                    f"val_auc: {logs.get('val_auc', 0):.4f} - "
                    f"val_loss: {logs.get('val_loss', 0):.4f}"
                )

        self.logger.info("\n Iniciando busca...")

        try:
            if search_hyperparameters:
                tuner.search(
                    train_gen,
                    validation_data=val_gen,
                    epochs=20,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1
                )

            self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            self.logger.info("\n Busca de hiperparâmetros concluída!")
            self.logger.info("\n MELHORES HIPERPARÂMETROS ENCONTRADOS:")
            self.logger.info(f"  - Dropout Rate: {self.best_hps.get('dropout_rate'):.3f}")
            self.logger.info(f"  - Dense Units: {self.best_hps.get('dense_units')}")
            self.logger.info(f"  - L2 Regularization: {self.best_hps.get('l2_reg'):.6f}")
            self.logger.info(f"  - Learning Rate: {self.best_hps.get('learning_rate'):.6f}")

            best_hps_dict = {
                'dropout_rate': float(self.best_hps.get('dropout_rate')),
                'dense_units': int(self.best_hps.get('dense_units')),
                'l2_reg': float(self.best_hps.get('l2_reg')),
                'learning_rate': float(self.best_hps.get('learning_rate'))
            }

            with open(os.path.join(self.config.RESULTS_DIR, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_hps_dict, f, indent=4)

            return self.best_hps

        except Exception as e:
            self.logger.error(f"X Erro durante busca de hiperparâmetros: {str(e)}")
            raise


# ============================================================================
# CROSS-VALIDATION MANAGER
# ============================================================================

class CrossValidationManager:
    """Gerencia o processo de Cross-Validation."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.fold_results = []

    def train_with_cross_validation(
            self,
            df: pd.DataFrame,
            model_builder: ModelBuilder,
            hyperparameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Executa treinamento com K-Fold Cross-Validation."""

        self.logger.info("=" * 60)
        self.logger.info(f"INICIANDO {self.config.N_FOLDS}-FOLD CROSS-VALIDATION")
        self.logger.info("=" * 60)

        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )

        # Preparar arrays para CV
        X = df['image_path'].values
        y = (df['label'] == 'malignant').astype(int).values

        fold_metrics = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'sensitivity': [],
            'specificity': []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f" FOLD {fold}/{self.config.N_FOLDS}")
            self.logger.info("=" * 60)

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            self.logger.info(f"Train: {len(train_df)} amostras "
                             f"(Benign: {(train_df['label'] == 'benign').sum()}, "
                             f"Malignant: {(train_df['label'] == 'malignant').sum()})")
            self.logger.info(f"Val: {len(val_df)} amostras "
                             f"(Benign: {(val_df['label'] == 'benign').sum()}, "
                             f"Malignant: {(val_df['label'] == 'malignant').sum()})")

            data_manager = DataManager(self.config, self.logger)
            train_gen, val_gen = data_manager.create_data_generators(train_df, val_df)

            class_weight = model_builder.calculate_class_weights(train_gen)

            if hyperparameters:
                model, base_model = model_builder.build_model(
                    dropout_rate=hyperparameters['dropout_rate'],
                    dense_units=hyperparameters['dense_units'],
                    l2_reg=hyperparameters['l2_reg'],
                    learning_rate=hyperparameters['learning_rate']
                )
            else:
                model, base_model = model_builder.build_model()

            fold_model_path = os.path.join(
                self.config.MODELS_DIR,
                f'fold_{fold}_best_model.keras'
            )

            callbacks = [
                EarlyStopping(
                    monitor='val_auc', # --> mudar para val_recall ??
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
                    fold_model_path,
                    monitor='val_auc', # --> mudar para val_recall ??
                    mode='max',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Treinar - Fase 1: Transfer Learning
            self.logger.info("\n Fase 1: Transfer Learning")
            history1 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=min(self.config.EPOCHS, self.config.FINE_TUNE_AT),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )

            # Fase 2: Fine-tuning (se necessário)
            if self.config.EPOCHS > self.config.FINE_TUNE_AT:
                self.logger.info("\n Fase 2: Fine-Tuning")

                base_model.trainable = True
                for layer in base_model.layers[:-self.config.FINE_TUNE_LAYERS]:
                    layer.trainable = False

                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss='binary_crossentropy',
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
                    epochs=self.config.EPOCHS,
                    initial_epoch=len(history1.history['loss']), # -> usar o loss faz sentido ?
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1
                )

            self.logger.info(f"\n Avaliando Fold {fold}...")
            best_model = keras.models.load_model(fold_model_path)
            y_pred_proba = best_model.predict(val_gen, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = val_gen.classes

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = sensitivity
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])

            fold_metrics['accuracy'].append(accuracy)
            fold_metrics['auc'].append(auc_score)
            fold_metrics['precision'].append(precision)
            fold_metrics['recall'].append(recall)
            fold_metrics['sensitivity'].append(sensitivity)
            fold_metrics['specificity'].append(specificity)

            self.logger.info(f"\n Métricas Fold {fold}:")
            self.logger.info(f"   Accuracy: {accuracy:.4f}")
            self.logger.info(f"   AUC: {auc_score:.4f}")
            self.logger.info(f"   Sensitivity: {sensitivity:.4f}")
            self.logger.info(f"   Specificity: {specificity:.4f}")
            self.logger.info(f"   Precision: {precision:.4f}")
            self.logger.info(f"   Recall: {recall:.4f}")

            self.fold_results.append({
                'fold': fold,
                'metrics': {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'recall': recall
                },
                'confusion_matrix': cm.tolist()
            })

        self.logger.info("\n" + "=" * 60)
        self.logger.info(" RESULTADOS AGREGADOS DO CROSS-VALIDATION")
        self.logger.info("=" * 60)

        aggregated_results = {}
        for metric_name, values in fold_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            aggregated_results[metric_name] = {
                'mean': mean,
                'std': std,
                'values': values
            }
            self.logger.info(f"{metric_name.capitalize():15s}: {mean:.4f} ± {std:.4f}")

        results_path = os.path.join(self.config.RESULTS_DIR, 'cv_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'fold_results': self.fold_results,
                'aggregated_results': {
                    k: {'mean': float(v['mean']), 'std': float(v['std'])}
                    for k, v in aggregated_results.items()
                }
            }, f, indent=4)

        self.logger.info(f"\n Resultados salvos em: {results_path}")

        return aggregated_results


# ============================================================================
# VISUALIZATION MANAGER
# ============================================================================

class VisualizationManager:
    """Gerencia a criação de visualizações."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def plot_cv_results(self, cv_results: Dict[str, Any]):
        """Plota os resultados do Cross-Validation."""

        self.logger.info("\n Gerando visualizações do Cross-Validation...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Resultados do Cross-Validation', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision', 'recall']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = cv_results[metric]['values']
            mean = cv_results[metric]['mean']
            std = cv_results[metric]['std']

            ax.bar(range(1, len(values) + 1), values, alpha=0.7, color='steelblue')
            ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Média: {mean:.3f}')
            ax.fill_between(
                range(1, len(values) + 1),
                mean - std,
                mean + std,
                alpha=0.2,
                color='red',
                label=f'Std: ±{std:.3f}'
            )
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()}: {mean:.3f} ± {std:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, len(values) + 1))

        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_DIR, 'cv_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f" Gráfico salvo: {save_path}")
        plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class SkinCancerPipeline:
    """Pipeline principal do sistema de classificação."""

    def __init__(self, config: Config):
        self.config = config
        self.config.create_directories()

        # Setup logger
        log_file = os.path.join(
            self.config.LOGS_DIR,
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        self.logger = LoggerConfig.setup_logger('SkinCancerPipeline', log_file)

        self.logger.info("=" * 60)
        self.logger.info(" SISTEMA DE CLASSIFICAÇÃO DE CÂNCER DE PELE")
        self.logger.info("=" * 60)
        self.logger.info(f"TensorFlow version: {tf.__version__}")
        self.logger.info(f"Keras Tuner disponível: Sim")

    def run(self, use_tuner: bool = True, use_cv: bool = True, search_hyperparameters: bool = True):
        """Executa o pipeline completo."""

        data_manager = DataManager(self.config, self.logger)
        df = data_manager.load_and_prepare_data()
        df = data_manager.balance_dataset(df)

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=self.config.RANDOM_SEED
        )

        if use_tuner:
            train_gen, val_gen = data_manager.create_data_generators(train_df, val_df)
            model_builder = ModelBuilder(self.config, self.logger)
            class_weight = model_builder.calculate_class_weights(train_gen)

            tuner = HyperparameterTuner(self.config, self.logger)
            best_hps = tuner.search_hyperparameters(train_gen, val_gen, class_weight, search_hyperparameters)

            hyperparameters = {
                'dropout_rate': best_hps.get('dropout_rate'),
                'dense_units': best_hps.get('dense_units'),
                'l2_reg': best_hps.get('l2_reg'),
                'learning_rate': best_hps.get('learning_rate')
            }
        else:
            hyperparameters = None

        if use_cv:
            model_builder = ModelBuilder(self.config, self.logger)
            cv_manager = CrossValidationManager(self.config, self.logger)
            cv_results = cv_manager.train_with_cross_validation(
                train_df,
                model_builder,
                hyperparameters
            )

            viz_manager = VisualizationManager(self.config, self.logger)
            viz_manager.plot_cv_results(cv_results)

        self.logger.info("\n" + "=" * 60)
        self.logger.info(" PIPELINE CONCLUÍDO COM SUCESSO!")
        self.logger.info("=" * 60)


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Configurar seeds para reprodutibilidade
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)

    # Criar e executar pipeline
    pipeline = SkinCancerPipeline(Config)

    # Executar com Tuner e Cross-Validation
    # Para desabilitar algum, use: use_tuner=False ou use_cv=False

    # Para rodar pipeline completa
    # pipeline.run(use_tuner=True, use_cv=True)

    # Para usar os melhores hiperparâmetros já encontrados
    pipeline.run(use_tuner=True, use_cv=True, search_hyperparameters=False)
