import os
import warnings
import logging

# Reduce TensorFlow C++/native logs (0=all, 1=INFO, 2=WARNING, 3=ERROR)
# Set before importing TensorFlow so native logs are filtered early.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress noisy Keras/UserWarning messages we don't want in normal logs.
# Filter the specific softmax warning emitted from keras.nn ops and
# ignore other UserWarnings originating from Keras modules.
# warnings.filterwarnings("ignore", message=".*softmax over axis -1.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"keras.*")

# Reduce verbosity from absl (used by TensorFlow) and from the tensorflow logger.
logging.getLogger('absl').setLevel(logging.ERROR)

from pandas import DataFrame, concat
import numpy
from scipy.signal import butter, filtfilt
from tensorflow import keras

# After TensorFlow is imported, ensure its Python logger is quiet.
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class Inferer:
    """Manages the inference model and performs classification on PPG data."""

    def __init__(self, model_path: str):
        """Initializes the Inferer with the specified model path."""
        self.model_path: str = model_path
        self.data: DataFrame = None

    def classify(self, data) -> dict:
        """Classify PPG data using the loaded model."""
        self.__add_data__(data)
        if len(self.data) != 250:
            print(f"Insufficient data for classification: {len(self.data)} samples (need 250).")
            return None

        return classify(self.data, self.model_path)


    def __add_data__(self, data) -> None:
        """Adds new PPG data for inference."""
        if self.data is None:
            self.data = data.copy()
            return

        self.data = concat([self.data, data]).tail(250)


def classify(data: DataFrame, model_path: str) -> dict[str, dict[str, object]]:
    """Classify PPG data and return per-channel results.

    Args:
        data (pandas.DataFrame): DataFrame containing the PPG channels as columns.
            Must be 10 seconds at 25 Hz (exactly 250 rows).
        model_path (str): Filesystem path to a Keras model file compatible with the
            network used for inference. Model must accept input shape (1, 250, 1).

    Returns:
        results: A mapping from channel name (e.g. ``"RED"``,
        ``"GREEN"``, ``"IR"``) to a result dictionary with the following keys:

        {
            "original_signal": numpy.ndarray, # 1-D original signal (shape (250,)), dtype float32
            "preprocessed_signal": numpy.ndarray, # 1-D preprocessed signal (shape (250,)), dtype float32
            "label": str, # classification label: 'SR' (sinusal) or 'AF' (fibrilación)
            "confidence": float # confidence score in [0.0, 1.0]
        }

    Example:
        >>> results = classify(df, "models/model.keras")
        >>> results["GREEN"]["label"]
        'SR'
        >>> results["RED"]["signal"].shape
        (250,)

    Raises:
        ValueError: If the input data does not have the expected frequency or length.
    """

    # Validate data frequency and length
    try:
        # Assuming the index is a timestamp in milliseconds
        first = int(data.iloc[0].name) 
        second = int(data.iloc[1].name)
        diff = second - first
        freq = 1.0 / (diff / 1000.0)  # ms to s
        if (abs(freq - 25.0) > 0.1):
            raise ValueError(f"Data frequency is {freq:.2f} Hz, expected 25.0 Hz.")
    except Exception as e:
        raise ValueError(f"Could not determine data frequency: {e}")

    if len(data) != 250:
        raise ValueError(f"Data length is {len(data)} samples, expected exactly 250 samples (10 seconds at 25 Hz).")

    # Preprocess each channel
    # The bandpass filter is expected to remove baseline wander and high-frequency noise, leaving the relevant cardiac components.
    # The robust normalization centers the signal around zero and scales it based on the median absolute deviation.
    processed: dict[str, numpy.ndarray] = {}
    for key in data.columns:
        signal = data[key].values.astype(numpy.float32)
        signal = bandpass_filter(signal, 0.5, 8.0, 25.0)
        signal = robust_normalize(signal)
        processed[key] = signal

    # Load model and classify each channel
    model = keras.models.load_model(model_path, compile=False)
    results: dict[str, dict[str, object]] = {}
    for key in processed:
        seg = processed[key].reshape(1, 250, 1)

        pred = model.predict(seg, verbose=0)[0]
        idx = int(numpy.argmax(pred))
        label = "SR" if idx == 0 else "AF"
        conf = float(numpy.max(pred))

        results[key] = {
            "original_signal": data[key].values.astype(numpy.float32),
            "preprocessed_signal": processed[key],
            "label": label,
            "confidence": conf
        }

    return results
    

def bandpass_filter(x: numpy.ndarray, lowcut: float, highcut: float, fs: float) -> numpy.ndarray:
    """Applies a Butterworth bandpass filter to the input signal x."""
    nyq = fs * 0.5
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)

def robust_normalize(x: numpy.ndarray) -> numpy.ndarray:
    """Applies robust normalization to the input signal x."""
    x = numpy.asarray(x, dtype=numpy.float32)
    med = numpy.median(x)
    mad = numpy.median(numpy.abs(x - med)) + 1e-8
    return (x - med) / mad
