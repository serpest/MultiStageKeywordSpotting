import queue
import time
import sounddevice as sd
import numpy as np
import tensorflow as tf
import keras
from keras import layers


KEYWORDS_10 = ['up', 'down', 'left', 'right', 'yes', 'no', 'on', 'off', 'go', 'stop']
SILENCE_KEYWORD = '_silence_'
UNKNOWN_KEYWORD = '_unknown_'

KEYWORDS_35 = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
    'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up',
    'visual', 'wow', 'yes', 'zero'
]


def get_keywords(mode: str) -> list[str]:
    if mode == '12':
        return KEYWORDS_10 + [SILENCE_KEYWORD, UNKNOWN_KEYWORD]
    elif mode == '35':
        return KEYWORDS_35
    else:
        raise ValueError("mode must be '12' or '35'")


class MFCCExtractor:

    def __init__(
        self,
        sample_rate: int = 16000,
        window_length_ms: int = 30,
        window_stride_ms: int = 10,
        num_mfcc: int = 40
    ) -> None:
        self.sample_rate = sample_rate
        self.window_length = int(sample_rate * window_length_ms / 1000)
        self.window_stride = int(sample_rate * window_stride_ms / 1000)
        self.num_mfcc = num_mfcc

    def __call__(self, audio: tf.Tensor) -> tf.Tensor:
        stft = tf.signal.stft(audio, frame_length=self.window_length, frame_step=self.window_stride)
        spectrogram = tf.abs(stft)
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=2 * self.num_mfcc,
            num_spectrogram_bins= tf.shape(stft)[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )
        mel_spectrograms = spectrogram @ linear_to_mel_weight_matrix
        mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfcc = mfcc[:, :self.num_mfcc]
        return mfcc


class EmbeddingBlockKWT(keras.Layer):

    def __init__(self, d: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d = d # Embedding dimension

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'d': self.d})
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        T = input_shape[1]
        F = input_shape[2]
        self.W_0 = self.add_weight(shape=(F, self.d))
        self.X_class = self.add_weight(shape=(1, 1, self.d))
        self.X_pos = self.add_weight(shape=(1, T + 1, self.d))

    def call(self, mfccs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(mfccs)[0]
        X_class_tiled = tf.tile(self.X_class, [batch_size, 1, 1]) # Shape: (batch_size, 1, d)
        # X_pos_tiled = tf.tile(self.X_pos, [batch_size, 1, 1]) # Shape: (batch_size, T + 1, d)
        X_1 = mfccs @ self.W_0 # Shape: (batch_size, T, d)
        X_0 = tf.concat([X_class_tiled, X_1], axis=1) + self.X_pos # Shape: (batch_size, T + 1, d)
        return X_0


class TransformerBlockKWT(keras.Layer):

    def __init__(self, k: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k = k # Number of attention heads
        self.d = 64 * k # Embedding dimension
        self.d_h = 64 # Dimension of each attention head
        self.msa = layers.MultiHeadAttention(
            num_heads=self.k,
            key_dim=self.d_h,
            use_bias=False
        )
        self.ln1 = layers.LayerNormalization()
        self.mlp = keras.Sequential([
            layers.Dense(4 * self.d, activation='gelu'),
            layers.Dense(self.d)
        ])
        self.ln2 = layers.LayerNormalization()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'k': self.k})
        return config

    def call(self, embeddings: tf.Tensor, training: bool = None) -> tf.Tensor:
        msa_out = self.msa(embeddings, embeddings, embeddings, training=training)
        x_tilde = self.ln1(msa_out + embeddings)
        mlp_out = self.mlp(x_tilde, training=training)
        x = self.ln2(mlp_out + x_tilde)
        return x


class KeywordTransformer(keras.Model):

    def __init__(self, num_classes: int, k: int, depth: int = 12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.k = k
        self.depth = depth
        self.d = 64 * k
        self.embedding_block = EmbeddingBlockKWT(d=self.d)
        self.transformer_blocks = [TransformerBlockKWT(k=k) for _ in range(depth)]
        self.head = layers.Dense(num_classes)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'k': self.k,
            'depth': self.depth
        })
        return config

    def call(self, mfccs: tf.Tensor, training: bool = None) -> tf.Tensor:
        x = self.embedding_block(mfccs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = x[:, 0, :] # Extract the class token
        logits = self.head(x)
        return logits


class TwoStageKeywordTransformer(keras.Model):

    def __init__(
        self,
        trigger_model: KeywordTransformer,
        keyword_model: KeywordTransformer,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.trigger_model = trigger_model
        self.keyword_model = keyword_model

    def call(self, mfccs: tf.Tensor, training: bool = None) -> tf.Tensor:
        trigger_logits = self.trigger_model(mfccs, training=training)  # Shape: (batch_size, 1)
        trigger_probs = tf.sigmoid(trigger_logits)
        keyword_logits = self.keyword_model(mfccs, training=training)  # Shape: (batch_size, num_classes - 1)
        keyword_probs = tf.nn.softmax(keyword_logits, axis=-1)
        weighted_keyword_probs = keyword_probs * trigger_probs
        unknown_probs = 1.0 - trigger_probs
        output_probs = tf.concat([weighted_keyword_probs, unknown_probs], axis=-1)  # Shape: (batch_size, num_classes)
        return output_probs


def main(
    detection_interval = 0.25,
    min_rms: float = 0.005,
    trigger_threshold: float = 0.6,
    prob_threshold: float = 0.75,
    min_interval_repeated_keyword: float = 1.0
) -> None:
    sample_rate = 16000
    window_duration = 1.0
    window_samples = int(sample_rate * window_duration)
    chunk_samples = int(sample_rate * detection_interval)
    keywords = get_keywords('35')
    mfcc_extractor = MFCCExtractor()
    trigger_model = keras.models.load_model(
        f'models/kwt_1_trigger_12.keras',
        custom_objects={'KeywordTransformer': KeywordTransformer},
        compile=False
    )
    keyword_model = keras.models.load_model(
        f'models/kwt_1_12.keras',
        custom_objects={'KeywordTransformer': KeywordTransformer},
        compile=False
    )
    trigger_model(tf.zeros([1, 98, 40], dtype=tf.float32), training=False) # Warm up the model
    keyword_model(tf.zeros([1, 98, 40], dtype=tf.float32), training=False) # Warm up the model
    chunk_queue = queue.Queue()
    audio_buffer = np.zeros(window_samples, dtype=np.float32)
    last_detected_keyword = None
    last_detected_time = 0.0
    print('Listening...')
    print('Press Ctrl+C to stop.')
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=chunk_samples,
            callback=lambda data, *_: chunk_queue.put(data[:, 0].copy())
        ):
            while True:
                chunk = chunk_queue.get()
                audio_buffer = np.concatenate([audio_buffer[chunk_samples:], chunk]) # Shift the buffer and append the new chunk
                rms = float(np.sqrt(np.mean(audio_buffer ** 2))) # Root Mean Square (RMS)
                if rms < min_rms: # Skip too quiet audio
                    continue
                mfcc = tf.expand_dims(mfcc_extractor(audio_buffer), axis=0)
                trigger_logits = trigger_model(mfcc, training=False)
                trigger_probs = tf.nn.softmax(trigger_logits, axis=-1).numpy()[0]
                if trigger_probs < trigger_threshold:
                    continue
                logits = keyword_model(mfcc, training=False)
                probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                top_index = int(np.argmax(probs))
                top_keyword = keywords[top_index]
                top_prob = float(probs[top_index])
                if (top_prob >= prob_threshold and top_keyword not in {SILENCE_KEYWORD, UNKNOWN_KEYWORD}):
                    current_time = time.time()
                    if last_detected_keyword != top_keyword or (current_time - last_detected_time) >= min_interval_repeated_keyword:
                        print(f'> {top_keyword} ({top_prob:.2f})')
                        last_detected_keyword = top_keyword
                        last_detected_time = current_time
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
