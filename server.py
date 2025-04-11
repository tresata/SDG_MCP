from pathlib import Path
import  os
from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic
from typing import Hashable
from sdv.metadata import Metadata

from mcp.server.fastmcp import FastMCP
from pandas.core.interchange.dataframe_protocol import DataFrame
import pandas as pd

mcp = FastMCP("SDG-MCP-SERVER")

# Optional: set max columns and width to display full DataFrame
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Disable line wrapping


@mcp.tool()
def read_schema(FilePath : str) -> dict[Hashable, str]:
    """
    Reads a CSV file and returns its schema in dictionary format.

    Parameters:
        FilePath (str): The path to the CSV file.

    Returns:
        dict: A dictionary where each key is a column name and each value is the
              corresponding data type as a string (e.g., 'int64', 'float64', 'object').
    """
    df = pd.read_csv(FilePath)
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return schema



@mcp.tool()
def read_top_5_reccords(FilePath: str) -> pd.DataFrame:
    """
    Reads a CSV file from the specified path and returns the top 5 records.

    Parameters:
        FilePath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the first 5 rows of the file.
    """
    return pd.read_csv(FilePath).head(5)


@mcp.tool()
def execute_python_script(script: str) -> str:
    """
    Executes a Python script provided as a string.

    Parameters:
        script (str): The Python code to be executed.

    Returns:
        str: "Success" if the script executes without errors.
             Otherwise, returns "Failure" followed by the error message.
    """
    try:
        exec(script)
        return "Success"
    except Exception as e:
        return f"Failure: {str(e)}"




@mcp.tool()
def generate_synthetic_data_from_gan(FilePath: str) -> str:
    """
    Generates synthetic tabular data using a CTGAN model trained on the input data.

    The function reads the dataset from the provided file path, trains a CTGAN model
    to learn the data distribution, generates synthetic samples, and saves them to a new
    file named 'Synthetic_{original_filename}.csv' in the same directory.

    Parameters:
        FilePath (str): The path to the CSV file containing the original data.

    Returns:Expected type 'Metadata', got 'SingleTableMetadata' instead
        str: "Success" if synthetic data was generated and saved successfully.
             Otherwise, returns "Failure" with an error message.
    """
    try:
        # Load data
        df = pd.read_csv(FilePath)
        #load meta
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        # Train CTGAN model

        model = CTGANSynthesizer(metadata=metadata)
        model.fit(df)

        # Generate synthetic data (same number of rows as original)
        synthetic_data = model.sample(len(df))

        # Create output filename
        dir_name, base_name = os.path.split(FilePath)
        synthetic_file_name = f"Synthetic_{base_name}"
        output_path = os.path.join(dir_name, synthetic_file_name)

        # Save synthetic data
        synthetic_data.to_csv(output_path, index=False)

        return "Success"
    except Exception as e:
        return f"Failure: {str(e)}"

@mcp.tool()
def profile_data(FilePath: str) -> dict:
    """
    Profiles the dataset and returns statistics such as null counts,
    unique values, min/max for numeric types.

    Parameters:
        FilePath (str): Path to the dataset (CSV).

    Returns:
        dict: Profiled statistics per column.
    """
    df = pd.read_csv(FilePath)
    profile = {}
    for col in df.columns:
        profile[col] = {
            "dtype": str(df[col].dtype),
            "nulls": df[col].isnull().sum(),
            "unique": df[col].nunique(),
            "min": df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
            "max": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
        }
    return profile




@mcp.tool()
def generate_fake_rows(columns: list[str], num_rows: int = 10) -> list[dict]:
    """
    Generates fake rows for a given list of columns using the Faker library.

    Parameters:
        columns (list): Column names.
        num_rows (int): Number of rows to generate.

    Returns:
        list of dicts: Each dict represents a row.
    """
    fake = Faker()
    data = []
    for _ in range(num_rows):
        row = {col: fake.word() for col in columns}
        data.append(row)
    return data


from sklearn.utils import resample
@mcp.tool()
def balance_classes(FilePath: str, label_column: str, strategy: str = 'over') -> str:
    """
    Balances class distribution via over/under sampling and saves the result.

    Parameters:
        FilePath (str): Path to CSV.
        label_column (str): Target column.
        strategy (str): 'over' or 'under'.

    Returns:
        str: Path to the resampled CSV file.
    """
    import pandas as pd
    from sklearn.utils import resample

    df = pd.read_csv(FilePath)
    majority_class = df[label_column].value_counts().idxmax()
    minority_class = df[label_column].value_counts().idxmin()

    df_major = df[df[label_column] == majority_class]
    df_minor = df[df[label_column] == minority_class]

    if strategy == 'over':
        df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major))
        balanced_df = pd.concat([df_major, df_minor_up])
    else:
        df_major_down = resample(df_major, replace=False, n_samples=len(df_minor))
        balanced_df = pd.concat([df_major_down, df_minor])

    output_path = "balanced_output.csv"
    balanced_df.to_csv(output_path, index=False)
    return output_path


@mcp.tool()
def generate_timegan_synthetic_data(
    real_data_path: str,
    sequence_length: int = 24,
    epochs: int = 500,
    batch_size: int = 128,
    latent_dim: int = 16
) -> str:
    """
    Trains a TimeGAN model on the provided real time series data and returns the path to a generated synthetic dataset.

    Parameters:
        real_data_path (str): Path to the CSV file containing real time series data.
        sequence_length (int): Length of time series sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Size of mini-batches.
        latent_dim (int): Dimensionality of the latent space.

    Returns:
        str: File path of the generated synthetic dataset in CSV format.
    """
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler

    # Load and preprocess data
    raw_data = pd.read_csv(real_data_path)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(raw_data.values)

    # Convert to sequences
    def create_sequences(data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
        return np.array(sequences)

    seq_data = create_sequences(data, sequence_length)

    # Define generator
    def build_generator():
        model = tf.keras.Sequential([
            layers.Input(shape=(sequence_length, latent_dim)),
            layers.LSTM(128, return_sequences=True),
            layers.TimeDistributed(layers.Dense(data.shape[1]))
        ])
        return model

    # Define discriminator
    def build_discriminator():
        model = tf.keras.Sequential([
            layers.Input(shape=(sequence_length, data.shape[1])),
            layers.LSTM(128, return_sequences=False),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    generator = build_generator()
    discriminator = build_discriminator()

    # Optimizers
    g_opt = tf.keras.optimizers.Adam(1e-4)
    d_opt = tf.keras.optimizers.Adam(1e-4)

    # Training loop
    for epoch in range(epochs):
        idx = np.random.randint(0, seq_data.shape[0], batch_size)
        real_batch = seq_data[idx]

        noise = np.random.normal(size=(batch_size, sequence_length, latent_dim))
        generated = generator.predict(noise)

        with tf.GradientTape() as tape:
            real_logits = discriminator(real_batch)
            fake_logits = discriminator(generated)
            d_loss = -tf.reduce_mean(tf.math.log(real_logits + 1e-8) + tf.math.log(1 - fake_logits + 1e-8))
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            noise = np.random.normal(size=(batch_size, sequence_length, latent_dim))
            fake = generator(noise)
            fake_logits = discriminator(fake)
            g_loss = -tf.reduce_mean(tf.math.log(fake_logits + 1e-8))
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))

    # Generate synthetic data
    final_noise = np.random.normal(size=(1000, sequence_length, latent_dim))
    synthetic = generator.predict(final_noise)

    # Inverse transform and save
    synthetic_reshaped = synthetic.reshape(-1, data.shape[1])
    synthetic_unscaled = scaler.inverse_transform(synthetic_reshaped)
    synthetic_df = pd.DataFrame(synthetic_unscaled, columns=raw_data.columns)

    output_path = "synthetic_timegan_output.csv"
    synthetic_df.to_csv(output_path, index=False)

    return output_path


from scipy.stats import ks_2samp

@mcp.tool()
def detect_column_drift(real_file: str, synth_file: str) -> dict[str, float]:
    """
    Detects statistical drift in each column between two datasets using KS test.

    Parameters:
        real_file (str): Path to real data.
        synth_file (str): Path to synthetic data.

    Returns:
        dict: p-values of drift tests per column.
    """
    real = pd.read_csv(real_file)
    synth = pd.read_csv(synth_file)
    pvals = {}
    for col in real.columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            _, p = ks_2samp(real[col], synth[col])
            pvals[col] = p
    return pvals




if __name__=="__main__":
    mcp.run(transport="stdio")

