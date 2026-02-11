"""
LSTM Fraud Detection Model
Deep learning for sequential transaction patterns
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. LSTM model will not be available.")


class AttentionLayer(layers.Layer):
    """Custom attention layer for LSTM"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return keras.backend.sum(output, axis=1)
    
    def get_config(self):
        return super().get_config()


class LSTMFraudDetector:
    """
    LSTM-based fraud detection model with optional attention
    """
    
    def __init__(self, seq_length: int, n_features: int):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def build_model(
        self, 
        lstm_units: List[int] = None,
        dropout: float = 0.3, 
        learning_rate: float = 0.001,
        use_attention: bool = False
    ) -> keras.Model:
        """
        Build Bidirectional LSTM architecture
        
        Args:
            lstm_units: List of LSTM units per layer
            dropout: Dropout rate
            learning_rate: Learning rate
            use_attention: Whether to use attention mechanism
        """
        lstm_units = lstm_units or [64, 32]
        
        if use_attention:
            return self._build_attention_model(lstm_units, dropout, learning_rate)
        
        model = keras.Sequential([
            # First LSTM layer
            layers.Bidirectional(
                layers.LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1),
                input_shape=(self.seq_length, self.n_features)
            ),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
        ])
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            model.add(layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_seq)
            ))
            model.add(layers.Dropout(dropout))
            model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout * 0.67))
        model.add(layers.Dense(16, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def _build_attention_model(
        self, 
        lstm_units: List[int], 
        dropout: float,
        learning_rate: float
    ) -> keras.Model:
        """Build LSTM with attention mechanism"""
        
        inputs = layers.Input(shape=(self.seq_length, self.n_features))
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(lstm_units[0], return_sequences=True)
        )(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        
        # Attention layer
        x = AttentionLayer()(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout * 0.67)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        epochs: int = 50, 
        batch_size: int = 256, 
        class_weight: Dict[int, float] = None,
        model_path: str = None
    ) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences (n_samples, seq_length, n_features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weight: Class weights for imbalanced data
            model_path: Path to save best model
        """
        if self.model is None:
            self.build_model()
        
        # Compute class weights if not provided
        if class_weight is None:
            n_fraud = y_train.sum()
            n_legit = len(y_train) - n_fraud
            class_weight = {
                0: 1.0,
                1: n_legit / n_fraud if n_fraud > 0 else 1.0
            }
        
        logger.info(f"Training LSTM with class weights: {class_weight}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
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
            )
        ]
        
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ))
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best metrics
        best_epoch = np.argmax(self.history.history['val_auc'])
        
        results = {
            'best_epoch': best_epoch,
            'train_auc': self.history.history['auc'][best_epoch],
            'val_auc': self.history.history['val_auc'][best_epoch],
            'train_loss': self.history.history['loss'][best_epoch],
            'val_loss': self.history.history['val_loss'][best_epoch]
        }
        
        logger.info(f"\nTraining Results:")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Training AUC: {results['train_auc']:.4f}")
        logger.info(f"  Validation AUC: {results['val_auc']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probability"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X, verbose=0).flatten()
    
    def predict_with_threshold(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction with custom threshold"""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Return training history"""
        if self.history is None:
            return {}
        return self.history.history
    
    def save(self, filepath: str):
        """Save model to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMFraudDetector':
        """Load model from disk"""
        model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        # Get shape from model
        input_shape = model.input_shape
        seq_length = input_shape[1]
        n_features = input_shape[2]
        
        detector = cls(seq_length, n_features)
        detector.model = model
        
        logger.info(f"Model loaded from {filepath}")
        return detector
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        return "Model not built yet"
